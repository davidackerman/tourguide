"""Lightweight sandboxing for analysis code.

This is a *best-effort* guardrail, not a security boundary. The threat model
is "an LLM wrote Python that accidentally shells out, reads files, or opens
sockets"; not "a determined attacker wants root on my container". For that
reason we use cheap checks:
    1. A regex-level deny-list for obvious danger (os/subprocess/pickle/etc).
    2. An import whitelist (evaluated against the user's source before it
       runs). Anything outside the list → rejected up front.
    3. We run the user code in a child process with resource limits so that
       even if something slips past (1)+(2), it can't wedge the worker.

Adapted from server/analysis_agent.py — same whitelists, trimmed to what's
relevant for the browser-originated requests the HF Space sees.
"""

from __future__ import annotations

import ast
import multiprocessing
import os
import re
import resource
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---- Policy -----------------------------------------------------------------

ALLOWED_IMPORTS = {
    "numpy",
    "np",
    "pandas",
    "pd",
    "scipy",
    "scipy.ndimage",
    "scipy.stats",
    "scipy.spatial",
    "scipy.signal",
    "skimage",
    "skimage.measure",
    "skimage.morphology",
    "skimage.filters",
    "skimage.transform",
    "skimage.segmentation",
    "matplotlib",
    "matplotlib.pyplot",
    "plt",
    # Seung-lab image-analysis stack — same domain as skimage/scipy.ndimage
    # but 10–100× faster on label volumes. Installed in the HF Dockerfile.
    "cc3d",
    "fastmorph",
    "fastremap",
    "edt",
    "kimimaro",
    "zmesh",
    "json",
    "math",
    "statistics",
    "itertools",
    "collections",
    "functools",
    "dataclasses",
    "typing",
    "re",
    "io",
    "base64",
    "traceback",
    "concurrent",
    "concurrent.futures",
    # Common stdlib utilities user/LLM code reaches for. None of these can
    # do anything dangerous on their own — file/socket/process modules
    # remain blocked elsewhere by the regex deny-list.
    "time",
    "datetime",
    "random",
    "copy",
    "string",
    "uuid",
    "warnings",
    "operator",
    "heapq",
    "bisect",
}

FORBIDDEN_PATTERNS = [
    # Process / shell
    (re.compile(r"\b__import__\s*\("), "dynamic __import__ is not allowed"),
    (re.compile(r"\beval\s*\("), "eval() is not allowed"),
    (re.compile(r"\bexec\s*\("), "exec() is not allowed"),
    (re.compile(r"\bcompile\s*\("), "compile() is not allowed"),
    (re.compile(r"subprocess"), "subprocess module is not allowed"),
    (re.compile(r"\bos\.(system|popen|exec[vl]p?e?)\b"), "shelling out via os is not allowed"),
    # Network / IO escape
    (re.compile(r"\bsocket\b"), "raw sockets are not allowed"),
    (re.compile(r"\brequests\b"), "outbound HTTP via requests is not allowed"),
    (re.compile(r"\burllib\b"), "outbound HTTP via urllib is not allowed"),
    (re.compile(r"\baiohttp\b"), "outbound HTTP via aiohttp is not allowed"),
    # Serialization bombs
    (re.compile(r"\bpickle\b"), "pickle is not allowed"),
    (re.compile(r"\bmarshal\b"), "marshal is not allowed"),
    # Obvious file ops outside /tmp
    (re.compile(r'\bopen\s*\(\s*["\'][^"\']*(?<!/tmp)[^"\']*["\']\s*,\s*["\']w'),
     "writing files outside /tmp is not allowed"),
]

# ---- Analysis ---------------------------------------------------------------


@dataclass
class SandboxVerdict:
    ok: bool
    reason: Optional[str] = None
    imports: List[str] = field(default_factory=list)


def check_code(source: str) -> SandboxVerdict:
    """Static-only check — no execution. Use before running."""
    for pat, msg in FORBIDDEN_PATTERNS:
        if pat.search(source):
            return SandboxVerdict(ok=False, reason=msg)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return SandboxVerdict(ok=False, reason=f"SyntaxError: {e.msg} at line {e.lineno}")

    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    disallowed = [m for m in imported if m not in ALLOWED_IMPORTS]
    if disallowed:
        return SandboxVerdict(
            ok=False,
            reason=f"imports not on the whitelist: {', '.join(sorted(set(disallowed)))}",
            imports=imported,
        )
    return SandboxVerdict(ok=True, imports=imported)


# ---- Execution --------------------------------------------------------------

# Limits applied to the worker process before user code runs.
# On the HF Space with 16 GB RAM and 2 vCPU, ops like binary_erosion or
# cc3d.connected_components on a 500M-voxel array take a couple of minutes.
# The 60s ceiling we started with was way too tight.
DEFAULT_TIMEOUT_S = 300  # 5 minutes wall-clock; user code should stay well under
DEFAULT_MEM_BYTES = 14 * 1024 * 1024 * 1024  # 14 GB — leaves headroom inside a 16 GB Space
DEFAULT_CPU_S = 600  # CPU-seconds; generous so wall-clock is the real gate


def _apply_rlimits(mem_bytes: int, cpu_s: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except (ValueError, OSError):
        pass


# Prelude executed in the sandbox before user code. Matches what the
# Pyodide worker pre-imports, so identical user code works in both places.
# The Seung-lab extras (cc3d, fastmorph, etc.) are optional imports — present
# on the HF Space, missing in Pyodide — so the prompt tells the LLM which
# set to use based on where the code is running.
#
# We also override builtins.print so user code's prints flow into
# _TG_STDOUT and end up in the response (and the modal's output area)
# instead of being lost in the subprocess. Without this the user has no
# visibility into anything their code printed during analysis.
_PRELUDE = """
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure as _sk_measure
# Optional Seung-lab imports — only the ones successfully installed.
try: import cc3d
except ImportError: cc3d = None
try: import fastmorph
except ImportError: fastmorph = None
try: import fastremap
except ImportError: fastremap = None
try: import edt
except ImportError: edt = None
try: import kimimaro
except ImportError: kimimaro = None
try: import zmesh
except ImportError: zmesh = None

# Print capture (mirrors the Pyodide worker setup).
import builtins as _b, sys as _sys
if not isinstance(_TG_STDOUT, list):
    _TG_STDOUT = []
# Subprocess stdout is inherited from the parent (uvicorn -> HF Container
# logs). But Python defaults to block buffering on a non-tty, so prints
# from a long-running analysis don't appear in the logs until the process
# exits. Force line buffering + flush=True so each print() is visible
# in HF Logs in real time.
try:
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
_real_print = _b.print
def _captured_print(*args, **kwargs):
    try:
        _TG_STDOUT.append(' '.join(str(a) for a in args))
    except Exception:
        pass
    kwargs.setdefault('flush', True)
    _real_print(*args, **kwargs)
_b.print = _captured_print
"""


def _worker_entrypoint(
    source: str,
    globals_dict: Dict[str, Any],
    out_queue: "multiprocessing.Queue",
    mem_bytes: int,
    cpu_s: int,
) -> None:
    _apply_rlimits(mem_bytes, cpu_s)
    os.environ["MPLBACKEND"] = "Agg"
    try:
        # Inject the prelude imports into the same globals dict the user
        # code sees. Without this, every analysis dies on 'NameError: np'
        # (or ndi/plt/pd/...).
        exec(compile(_PRELUDE, "<tourguide-prelude>", "exec"), globals_dict)
        exec(compile(source, "<tourguide-analysis>", "exec"), globals_dict)
        # Pull out the _TG_* outputs the user code set.
        result = {k: v for k, v in globals_dict.items() if k.startswith("_TG_")}
        out_queue.put(("ok", result))
    except BaseException as exc:  # noqa: BLE001 — we want every error
        out_queue.put(("error", {
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }))


@dataclass
class SandboxResult:
    ok: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    traceback: Optional[str] = None


_HEARTBEAT_S = 10
_log = __import__("logging").getLogger("tourguide")


def run_sandboxed(
    source: str,
    globals_dict: Dict[str, Any],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    mem_bytes: int = DEFAULT_MEM_BYTES,
    cpu_s: int = DEFAULT_CPU_S,
) -> SandboxResult:
    """Run `source` with `globals_dict` in a child process with rlimits.

    Note: multiprocessing pickles globals_dict to send it over. numpy arrays
    handle that fine; anything exotic (PyProxy, open file handles) would
    break — but we don't hand user code anything exotic.

    We poll the process with short joins instead of one long join, so we
    can emit a heartbeat to the parent's logger every ~10 s. That gives
    the user something to watch in HF Container Logs even when the AI's
    code is silent (no prints).
    """
    import time as _time

    verdict = check_code(source)
    if not verdict.ok:
        return SandboxResult(ok=False, error=verdict.reason)

    import queue as _queue

    ctx = multiprocessing.get_context("spawn")
    q: "multiprocessing.Queue" = ctx.Queue()
    proc = ctx.Process(
        target=_worker_entrypoint,
        args=(source, globals_dict, q, mem_bytes, cpu_s),
        daemon=True,
    )
    proc.start()
    start = _time.time()
    last_heartbeat = start
    payload_pair = None

    # Actively drain the queue while the worker runs. multiprocessing.Queue
    # is backed by a Pipe; if the worker tries to put a sizable result
    # (DataFrame, numpy array) and the parent is only doing proc.join,
    # the Pipe buffer fills and the worker blocks on put forever — a
    # classic deadlock that looked like "still running" via heartbeat
    # but the script was actually done.
    while True:
        try:
            payload_pair = q.get(timeout=1)
            break
        except _queue.Empty:
            pass
        elapsed = _time.time() - start
        if elapsed > timeout_s:
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
            return SandboxResult(ok=False, error=f"timed out after {timeout_s}s")
        if not proc.is_alive():
            # Subprocess exited without producing a result — shouldn't
            # happen in normal operation, but bail rather than spin.
            break
        if _time.time() - last_heartbeat >= _HEARTBEAT_S:
            last_heartbeat = _time.time()
            _log.info("sandbox heartbeat: still running, elapsed %.0fs / %ds", elapsed, timeout_s)

    # Reap the subprocess.
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)

    if payload_pair is None:
        if proc.exitcode != 0:
            return SandboxResult(ok=False, error=f"worker exited with code {proc.exitcode}")
        return SandboxResult(ok=False, error="worker produced no result")

    tag, payload = payload_pair
    if tag == "ok":
        return SandboxResult(ok=True, outputs=payload)
    return SandboxResult(ok=False, error=payload.get("message"), traceback=payload.get("traceback"))
