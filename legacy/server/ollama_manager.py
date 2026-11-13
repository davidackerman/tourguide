"""
Ollama Manager - Auto-start Ollama if not running

Ensures Ollama is available on cluster nodes or local machines.

Note: Ollama server is kept running to avoid reloading models into memory.
The server will persist until manually stopped or the cluster node is terminated.
"""

import subprocess
import time
import os
import sys
import atexit
import signal


class OllamaManager:
    """Manages Ollama server lifecycle."""

    def __init__(self, persist: bool = True):
        """
        Initialize Ollama manager.

        Args:
            persist: If True, keep Ollama running after script exits (default: True)
                     This avoids reloading models into memory on each run.
        """
        self.ollama_process = None
        self.started_by_us = False
        self.persist = persist

    def is_ollama_running(self) -> bool:
        """
        Check if Ollama is already running.

        Returns:
            True if Ollama is responding, False otherwise
        """
        try:
            # Try to run a simple ollama command
            result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    def _get_available_gpu(self) -> int:
        """
        Auto-detect an available GPU with low memory usage.

        Returns:
            GPU ID (0-7) or None if no GPUs available
        """
        try:
            # Run nvidia-smi to get GPU memory usage
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            # Parse output: "0, 1234" format
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    gpu_id = int(parts[0].strip())
                    mem_used = int(parts[1].strip())
                    gpus.append((gpu_id, mem_used))

            if not gpus:
                return None

            # Find GPU with least memory usage
            gpus.sort(key=lambda x: x[1])
            best_gpu = gpus[0]

            # Only use GPU if it has less than 10GB used (mostly free)
            if best_gpu[1] < 10000:
                return best_gpu[0]

            return None

        except Exception as e:
            print(f"[OLLAMA] Could not detect GPU: {e}", flush=True)
            return None

    def start_ollama(self, log_file: str = None, gpu_id: int = None) -> bool:
        """
        Start Ollama server in the background.

        Args:
            log_file: Path to log file (default: ~/ollama-serve.log)
            gpu_id: GPU ID to use (default: auto-detect available GPU)

        Returns:
            True if started successfully, False otherwise
        """
        if log_file is None:
            log_file = os.path.expanduser("~/ollama-serve.log")

        try:
            # Set up environment with GPU configuration
            env = os.environ.copy()

            # Check if CUDA_VISIBLE_DEVICES is already set (e.g., by cluster scheduler)
            existing_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

            if existing_cuda_devices is not None:
                # Respect existing GPU assignment from cluster/user
                print(
                    f"[OLLAMA] Using existing CUDA_VISIBLE_DEVICES={existing_cuda_devices}",
                    flush=True,
                )
                # IMPORTANT: When CUDA_VISIBLE_DEVICES is set by scheduler (e.g., LSF),
                # the visible GPUs are renumbered starting from 0.
                # So we don't need to change it, just use it as-is.
            elif gpu_id is not None:
                # Use specified GPU
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                print(f"[OLLAMA] Using GPU {gpu_id}", flush=True)
            else:
                # Auto-detect if no GPU specified and no existing assignment
                gpu_id = self._get_available_gpu()
                if gpu_id is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    print(f"[OLLAMA] Auto-detected available GPU: {gpu_id}", flush=True)
                else:
                    print(
                        "[OLLAMA] Warning: No GPU specified, using CPU mode", flush=True
                    )

            # Set Ollama-specific environment variables for better GPU memory management
            # Limit to 1 loaded model (only nemotron for queries; chatterbox TTS is via pip)
            env["OLLAMA_MAX_LOADED_MODELS"] = "1"
            # Set max queue size
            env["OLLAMA_MAX_QUEUE"] = "512"

            print(
                f"[OLLAMA] Ollama config: MAX_LOADED_MODELS=1",
                flush=True,
            )

            print(f"[OLLAMA] Starting Ollama server (logs: {log_file})...", flush=True)

            # Open log file
            log_handle = open(log_file, "a")

            # Start ollama serve in background
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent process group
                env=env,
            )

            self.started_by_us = True

            # Register cleanup on exit only if not persisting
            if not self.persist:
                atexit.register(self.cleanup)
                # Also handle SIGTERM and SIGINT
                signal.signal(signal.SIGTERM, self._signal_handler)
                signal.signal(signal.SIGINT, self._signal_handler)
                print(
                    "[OLLAMA] Registered cleanup handlers (Ollama will stop on exit)",
                    flush=True,
                )
            else:
                print(
                    "[OLLAMA] Persist mode enabled (Ollama will keep running after script exits)",
                    flush=True,
                )

            # Wait a bit for Ollama to start
            print("[OLLAMA] Waiting for Ollama to initialize...", flush=True)
            for i in range(10):  # Wait up to 10 seconds
                time.sleep(1)
                if self.is_ollama_running():
                    print(
                        f"[OLLAMA] Ollama started successfully (PID: {self.ollama_process.pid})",
                        flush=True,
                    )
                    if self.persist:
                        print(
                            "[OLLAMA] Note: Ollama will continue running to keep models in memory",
                            flush=True,
                        )

                    # Give Ollama an extra moment to fully initialize before model loading
                    print(
                        "[OLLAMA] Allowing Ollama extra time to stabilize...",
                        flush=True,
                    )
                    time.sleep(2)

                    return True

            print("[OLLAMA] Warning: Ollama may not have started properly", flush=True)
            return False

        except FileNotFoundError:
            print("[OLLAMA] Error: 'ollama' command not found in PATH", flush=True)
            print("[OLLAMA] Please ensure Ollama is installed", flush=True)
            return False
        except Exception as e:
            print(f"[OLLAMA] Error starting Ollama: {e}", flush=True)
            return False

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        print(f"\n[OLLAMA] Received signal {signum}, cleaning up...", flush=True)
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """
        Stop Ollama if we started it and persist=False.

        Note: By default (persist=True), Ollama keeps running to avoid
        reloading models into memory. It will be killed when the cluster
        node terminates or you manually stop it.
        """
        if not self.persist and self.started_by_us and self.ollama_process:
            try:
                print(
                    f"[OLLAMA] Stopping Ollama server (PID: {self.ollama_process.pid})...",
                    flush=True,
                )

                # Try graceful shutdown first
                self.ollama_process.terminate()

                # Wait up to 5 seconds for graceful shutdown
                try:
                    self.ollama_process.wait(timeout=5)
                    print("[OLLAMA] Ollama stopped gracefully", flush=True)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop
                    print("[OLLAMA] Force killing Ollama...", flush=True)
                    self.ollama_process.kill()
                    self.ollama_process.wait()
                    print("[OLLAMA] Ollama force stopped", flush=True)

            except Exception as e:
                print(f"[OLLAMA] Error during cleanup: {e}", flush=True)
            finally:
                self.ollama_process = None
                self.started_by_us = False
        elif self.persist and self.started_by_us:
            # In persist mode, just let the user know Ollama is still running
            pass  # Ollama continues running

    def ensure_running(self, log_file: str = None, gpu_id: int = None) -> bool:
        """
        Ensure Ollama is running, starting it if necessary.

        Args:
            log_file: Optional custom log file path
            gpu_id: Optional GPU ID to use (auto-detected if not specified)

        Returns:
            True if Ollama is running, False otherwise
        """
        if self.is_ollama_running():
            print("[OLLAMA] Ollama is already running", flush=True)
            return True

        print("[OLLAMA] Ollama not detected, attempting to start...", flush=True)
        return self.start_ollama(log_file, gpu_id)


# Global instance
_ollama_manager = None


def get_ollama_manager() -> OllamaManager:
    """Get or create global Ollama manager instance."""
    global _ollama_manager
    if _ollama_manager is None:
        _ollama_manager = OllamaManager()
    return _ollama_manager


def ensure_ollama_running(log_file: str = None, gpu_id: int = None) -> bool:
    """
    Convenience function to ensure Ollama is running.

    Args:
        log_file: Optional custom log file path
        gpu_id: Optional GPU ID to use (auto-detected if not specified)

    Returns:
        True if Ollama is running, False otherwise
    """
    manager = get_ollama_manager()
    return manager.ensure_running(log_file, gpu_id)


def preload_model(model_name: str = "nemotron-3-nano", timeout: int = 60) -> bool:
    """
    Preload a model into GPU memory to avoid cold-start issues.

    This sends a small test query to force the model to load into GPU memory.
    Subsequent queries will be faster and avoid CUDA allocation race conditions.

    Args:
        model_name: Name of the Ollama model to preload
        timeout: Maximum seconds to wait for preload (default: 60)

    Returns:
        True if preload successful, False otherwise
    """
    print(f"[OLLAMA] Preloading model '{model_name}' into GPU memory...", flush=True)

    try:
        import ollama

        # Send a minimal test query to force model loading
        start_time = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            options={"num_predict": 1},  # Generate only 1 token to speed up
        )

        elapsed = time.time() - start_time
        print(
            f"[OLLAMA] Model '{model_name}' preloaded successfully ({elapsed:.1f}s)",
            flush=True,
        )
        return True

    except Exception as e:
        print(
            f"[OLLAMA] Warning: Failed to preload model '{model_name}': {e}", flush=True
        )
        return False
