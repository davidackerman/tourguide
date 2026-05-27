"""Browser tunnel: read zarr chunks from the user's browser over a WebSocket.

Normal case: the browser has the zarr (e.g. a folder the user picked with
FileSystemDirectoryHandle). We can't reach that folder from the cloud, but
the browser's service worker can serve it. So the flow is:

    Browser                         HF Space
    -------                         --------
    opens /ws/bridge/<sid>  ←──→   accepts connection
                                    when it wants byte `path`:
                                      sends {type: "request", req_id, path}
    fetches path via SW,
    sends {type: "response",
           req_id, bytes_b64}   ──→ resolves the future keyed by req_id
                                   returns bytes to zarr loader

We expose the per-key fetch as a zarr-python v3 ``Store`` and let
zarr-python handle metadata parsing, codec decoding, and chunk
reassembly. v3 supports both zarr v2 and v3 transparently, plus
sharding and arbitrary v3 codec pipelines (zstd, crc32c, …) — the
previous hand-rolled v2-only loader is gone.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import zarr.api.asynchronous as zarr_async
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype

log = logging.getLogger("browser_store")


@dataclass
class PendingRequest:
    future: asyncio.Future
    path: str


class TunnelSession:
    """Holds state for a single browser WS connection.

    Each analysis request typically owns one session. The session maps
    request ids (minted server-side) to Futures resolved when the browser
    replies. All `read(path)` calls are awaited by the loader.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._next_id = 0
        self._pending: Dict[int, PendingRequest] = {}
        self._send: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
        self._closed = False
        self._cache: Dict[str, Optional[bytes]] = {}

    def attach(self, send_coro: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        self._send = send_coro

    def is_ready(self) -> bool:
        return self._send is not None and not self._closed

    async def read(self, path: str, *, timeout: float = 30.0) -> Optional[bytes]:
        """Fetch bytes for `path` from the browser. Returns None for 404."""
        if path in self._cache:
            return self._cache[path]
        if not self.is_ready():
            raise RuntimeError("tunnel not connected")
        req_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[Optional[bytes]] = loop.create_future()
        self._pending[req_id] = PendingRequest(future=fut, path=path)
        assert self._send is not None
        await self._send({"type": "request", "req_id": req_id, "path": path})
        try:
            data = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise
        self._cache[path] = data
        return data

    def resolve(self, req_id: int, data_b64: Optional[str], found: bool) -> None:
        pending = self._pending.pop(req_id, None)
        if pending is None:
            log.warning("tunnel response for unknown req_id=%s (session=%s)", req_id, self.session_id)
            return
        if not found:
            pending.future.set_result(None)
            return
        try:
            bytes_ = base64.b64decode(data_b64 or "") if data_b64 else b""
        except Exception as exc:  # noqa: BLE001
            pending.future.set_exception(exc)
            return
        pending.future.set_result(bytes_)

    def close(self, reason: str = "closed") -> None:
        self._closed = True
        for p in list(self._pending.values()):
            if not p.future.done():
                p.future.set_exception(RuntimeError(f"tunnel {reason}"))
        self._pending.clear()


# ----------------------------------------------------------------------------
# zarr-python v3 Store wrapping the tunnel's async read(path) callable.
# Supports v2 + v3 + sharding + arbitrary codec pipelines via zarr-python.
# ----------------------------------------------------------------------------


class TunnelStore(Store):
    """Read-only zarr-python v3 ``Store`` backed by an async fetch coroutine.

    The tunnel only exposes a single ``read(key) -> bytes | None`` async
    operation, so we don't pretend to support listing or writes. Range
    requests are honored by fetching the whole key and slicing — the
    tunnel protocol is per-key, not byte-range, so true HTTP-style ranges
    aren't possible without a protocol change. For sharded inputs this
    means an entire shard is materialized to extract one inner chunk;
    correct, just not optimal. Optimization is a follow-up.
    """

    def __init__(self, read: Callable[[str], Awaitable[Optional[bytes]]]) -> None:
        super().__init__(read_only=True)
        self._is_open = True
        self._read = read

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TunnelStore) and other._read is self._read

    def __hash__(self) -> int:
        return id(self._read)

    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_partial_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return False

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: Optional[ByteRequest] = None,
    ) -> Optional[Buffer]:
        raw = await self._read(key)
        if raw is None:
            return None
        if byte_range is None:
            data = raw
        elif isinstance(byte_range, RangeByteRequest):
            data = raw[byte_range.start : byte_range.end]
        elif isinstance(byte_range, OffsetByteRequest):
            data = raw[byte_range.offset :]
        elif isinstance(byte_range, SuffixByteRequest):
            data = raw[-byte_range.suffix :]
        else:
            raise TypeError(f"unsupported byte range type {type(byte_range).__name__}")
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[Any],
    ) -> "list[Optional[Buffer]]":
        out: "list[Optional[Buffer]]" = []
        for key, byte_range in key_ranges:
            out.append(await self.get(key, prototype, byte_range))
        return out

    async def exists(self, key: str) -> bool:
        return (await self._read(key)) is not None

    async def set(self, key: str, value: Buffer) -> None:  # pragma: no cover
        raise NotImplementedError("TunnelStore is read-only")

    async def delete(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError("TunnelStore is read-only")

    async def list(self) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError("TunnelStore does not support listing")
        yield ""  # makes this an async generator

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError("TunnelStore does not support listing")
        yield ""

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError("TunnelStore does not support listing")
        yield ""


async def read_zarr_scale(
    read: Callable[[str], Awaitable[Optional[bytes]]],
    scale_path: str,
    selection: Optional[Tuple[Any, ...]] = None,
) -> np.ndarray:
    """Open the array at ``scale_path`` via a TunnelStore and materialize as numpy.

    Works for both zarr v2 (``.zarray`` + chunk files) and zarr v3
    (``zarr.json``, sharded chunks, arbitrary codec pipelines) —
    zarr-python v3 auto-detects the format. ``selection`` is the indexing
    tuple passed to ``arr.getitem`` — pass ``Ellipsis`` (the default) to
    read everything, or a tuple of ``slice``/int for an ROI read so only
    the chunks intersecting the slice come back over the tunnel.
    """
    store = TunnelStore(read)
    arr = await zarr_async.open_array(store=store, path=scale_path or "")
    data = await arr.getitem(selection if selection is not None else Ellipsis)
    return np.asarray(data)
