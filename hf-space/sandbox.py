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
    "cc3d",
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
DEFAULT_TIMEOUT_S = 60
DEFAULT_MEM_BYTES = 14 * 1024 * 1024 * 1024  # 14 GB — leaves headroom inside a 16 GB Space
DEFAULT_CPU_S = 120  # CPU-seconds; generous so the wall-clock timeout is the real gate


def _apply_rlimits(mem_bytes: int, cpu_s: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except (ValueError, OSError):
        pass


def _worker_entrypoint(
    source: str,
    globals_dict: Dict[str, Any],
    out_queue: "multiprocessing.Queue",
    mem_bytes: int,
    cpu_s: int,
) -> None:
    _apply_rlimits(mem_bytes, cpu_s)
    # Discourage the user code from spawning its own subprocesses even if it
    # somehow got past the whitelist.
    os.environ["MPLBACKEND"] = "Agg"
    try:
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
    """
    verdict = check_code(source)
    if not verdict.ok:
        return SandboxResult(ok=False, error=verdict.reason)

    ctx = multiprocessing.get_context("spawn")
    queue: "multiprocessing.Queue" = ctx.Queue()
    proc = ctx.Process(
        target=_worker_entrypoint,
        args=(source, globals_dict, queue, mem_bytes, cpu_s),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return SandboxResult(ok=False, error=f"timed out after {timeout_s}s")

    if proc.exitcode != 0 and queue.empty():
        return SandboxResult(ok=False, error=f"worker exited with code {proc.exitcode}")

    try:
        tag, payload = queue.get_nowait()
    except Exception:  # noqa: BLE001
        return SandboxResult(ok=False, error="worker produced no result")

    if tag == "ok":
        return SandboxResult(ok=True, outputs=payload)
    return SandboxResult(ok=False, error=payload.get("message"), traceback=payload.get("traceback"))
