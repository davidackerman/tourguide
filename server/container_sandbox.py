"""
Unified Container Sandbox for Secure Code Execution.

Supports multiple container runtimes (Docker, Apptainer/Singularity) with
automatic detection and backend selection.
"""

import os
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional


class DockerBackend:
    """Docker backend for container execution (original implementation)."""

    IMAGE_NAME = "tourguide-analysis:latest"

    def __init__(
        self,
        db_path: str,
        timeout: int = 60,
        memory_limit: str = "512m",
        cpu_quota: int = 50000
    ):
        """
        Initialize Docker backend.

        Args:
            db_path: Path to organelle database
            timeout: Execution timeout in seconds (default: 60)
            memory_limit: Container memory limit (default: 512m)
            cpu_quota: CPU quota in microseconds per 100ms (default: 50000 = 50%)
        """
        self.db_path = Path(db_path).resolve()
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.results_base_dir = Path("analysis_results")

        # Ensure results directory exists
        self.results_base_dir.mkdir(exist_ok=True)

        # Build image if needed
        self._ensure_image()

        print(f"[DOCKER_BACKEND] Initialized (timeout={timeout}s, mem={memory_limit})")

    def _ensure_image(self):
        """Build Docker image if it doesn't exist."""
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", self.IMAGE_NAME],
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            print(f"[DOCKER_BACKEND] Using existing image: {self.IMAGE_NAME}")
            return

        print(f"[DOCKER_BACKEND] Building image: {self.IMAGE_NAME}")
        self._build_image()

    def _build_image(self):
        """Build the analysis Docker image."""
        dockerfile_content = """FROM python:3.11-slim

# Install build dependencies and Python packages
RUN apt-get update && \\
    apt-get install -y gcc g++ && \\
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
    pandas==2.1.4 \\
    numpy==1.24.4 \\
    plotly==5.18.0 \\
    matplotlib==3.8.2 \\
    seaborn==0.13.0 \\
    kaleido==0.2.1

# Create directories
RUN mkdir -p /data /output /code

# Create non-root user
RUN useradd -m -u 1000 analyst && \\
    chown -R analyst:analyst /output

# Switch to non-root user
USER analyst

WORKDIR /code

# Default command
CMD ["python", "analysis.py"]
"""

        # Write Dockerfile to temp location
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)

            # Build image
            try:
                subprocess.run(
                    ["docker", "build", "-t", self.IMAGE_NAME, temp_dir],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"[DOCKER_BACKEND] Successfully built image: {self.IMAGE_NAME}")
            except subprocess.CalledProcessError as e:
                print(f"[DOCKER_BACKEND] Failed to build image:")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                raise RuntimeError(f"Failed to build Docker image: {e.stderr}")

    def execute(self, code: str, session_id: str) -> Dict[str, Any]:
        """
        Execute Python code in isolated Docker container.

        Args:
            code: Python code to execute
            session_id: Unique session identifier

        Returns:
            Dictionary with execution results:
                - status: 'success', 'error', or 'timeout'
                - stdout: Standard output
                - stderr: Standard error
                - plots: List of generated plot filenames
                - output_path: Path to output directory
                - execution_time: Time taken in seconds
        """
        start_time = time.time()

        # Create session directory
        session_dir = self.results_base_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # Create code file
        code_file = session_dir / "analysis.py"
        code_file.write_text(code)

        # Run container
        try:
            result = self._run_container(session_dir)
            execution_time = time.time() - start_time

            # Extract plot files
            plot_files = self._get_plot_files(session_dir)

            # Save logs
            (session_dir / "stdout.log").write_text(result["stdout"])
            (session_dir / "stderr.log").write_text(result["stderr"])

            return {
                "status": result["status"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "plots": plot_files,
                "output_path": str(session_dir),
                "execution_time": execution_time
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"[DOCKER_BACKEND] Execution failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                "status": "error",
                "stdout": "",
                "stderr": f"Execution failed: {str(e)}",
                "plots": [],
                "output_path": str(session_dir),
                "execution_time": execution_time
            }

    def _run_container(self, session_dir: Path) -> Dict[str, Any]:
        """
        Run Docker container with code.

        Args:
            session_dir: Session directory containing code and output

        Returns:
            Dictionary with status, stdout, stderr
        """
        # Prepare volume mounts
        volumes = {
            str(self.db_path.parent): "/data:ro",  # Database directory (read-only)
            str(session_dir): "/code:ro",          # Code file (read-only)
            str(session_dir): "/output:rw"         # Output directory (read-write)
        }

        # Build docker run command
        docker_cmd = [
            "docker", "run",
            "--rm",                                 # Remove container after execution
            "--network", "none",                    # No network access
            "--memory", self.memory_limit,          # Memory limit
            "--cpu-quota", str(self.cpu_quota),     # CPU limit
            "--cpus", "0.5",                        # Alternative CPU limit
            "--cap-drop", "ALL",                    # Drop all capabilities
            "--security-opt", "no-new-privileges",  # Security hardening
        ]

        # Add volume mounts
        for host_path, container_mount in volumes.items():
            docker_cmd.extend(["-v", f"{host_path}:{container_mount}"])

        # Add image and command
        docker_cmd.extend([
            self.IMAGE_NAME,
            "python", "/code/analysis.py"
        ])

        print(f"[DOCKER_BACKEND] Running container with timeout={self.timeout}s")

        try:
            # Run with timeout
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            status = "success" if result.returncode == 0 else "error"

            return {
                "status": status,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            print(f"[DOCKER_BACKEND] Execution timed out after {self.timeout}s")

            # Try to stop the container (best effort)
            try:
                subprocess.run(
                    ["docker", "ps", "-q", "--filter", f"ancestor={self.IMAGE_NAME}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            except Exception:
                pass  # Ignore cleanup errors

            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Execution exceeded timeout of {self.timeout} seconds"
            }

        except Exception as e:
            print(f"[DOCKER_BACKEND] Container execution error: {e}")
            return {
                "status": "error",
                "stdout": "",
                "stderr": str(e)
            }

    def _get_plot_files(self, session_dir: Path) -> List[str]:
        """
        Get list of plot files generated in output directory.

        Args:
            session_dir: Session directory

        Returns:
            List of plot filenames (relative to session_dir)
        """
        plot_files = []

        # Look for common plot file extensions
        for ext in ['.html', '.png', '.jpg', '.jpeg', '.svg']:
            for file_path in session_dir.glob(f"*{ext}"):
                # Exclude the code file and log files
                if file_path.name not in ['analysis.py', 'stdout.log', 'stderr.log']:
                    plot_files.append(file_path.name)

        print(f"[DOCKER_BACKEND] Found {len(plot_files)} plot file(s): {plot_files}")
        return sorted(plot_files)


class ContainerSandbox:
    """
    Unified container sandbox with automatic runtime detection.

    Supports Docker and Apptainer/Singularity backends.
    """

    def __init__(
        self,
        db_path: str,
        timeout: int = 60,
        memory_limit: str = "512m",
        cpu_quota: int = 50000,
        runtime: Optional[str] = None
    ):
        """
        Initialize container sandbox with automatic backend selection.

        Args:
            db_path: Path to organelle database
            timeout: Execution timeout in seconds (default: 60)
            memory_limit: Container memory limit (default: 512m)
            cpu_quota: CPU quota (default: 50000)
            runtime: Force specific runtime ('docker', 'apptainer', 'singularity', or None for auto)
        """
        self.db_path = db_path
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota

        # Detect or use specified runtime
        if runtime and runtime.lower() != "auto":
            self.runtime = runtime.lower()
            if not self._check_runtime(self.runtime):
                raise RuntimeError(f"Specified runtime '{runtime}' is not available")
        else:
            # Auto-detect runtime
            runtime_from_env = os.environ.get("CONTAINER_RUNTIME", "auto").lower()
            if runtime_from_env != "auto":
                self.runtime = runtime_from_env
                if not self._check_runtime(self.runtime):
                    raise RuntimeError(f"Runtime '{runtime_from_env}' from CONTAINER_RUNTIME env var is not available")
            else:
                self.runtime = self._detect_runtime()

        # Initialize appropriate backend
        if self.runtime in ["apptainer", "singularity"]:
            from server.apptainer_sandbox import ApptainerBackend
            self.backend = ApptainerBackend(db_path, timeout, memory_limit, cpu_quota)
            print(f"[CONTAINER_SANDBOX] Using Apptainer backend")
        elif self.runtime == "docker":
            self.backend = DockerBackend(db_path, timeout, memory_limit, cpu_quota)
            print(f"[CONTAINER_SANDBOX] Using Docker backend")
        else:
            raise RuntimeError(f"Unsupported runtime: {self.runtime}")

    def _check_command(self, command: str) -> bool:
        """
        Check if a command is available.

        Args:
            command: Command name to check

        Returns:
            True if command is available, False otherwise
        """
        try:
            result = subprocess.run(
                [command, "version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_runtime(self, runtime: str) -> bool:
        """
        Check if a specific runtime is available.

        Args:
            runtime: Runtime name ('docker', 'apptainer', 'singularity')

        Returns:
            True if runtime is available, False otherwise
        """
        return self._check_command(runtime)

    def _detect_runtime(self) -> str:
        """
        Detect available container runtime.

        Priority: apptainer > singularity > docker

        Returns:
            Runtime name

        Raises:
            RuntimeError: If no container runtime is found
        """
        # Check for Apptainer first (cluster priority)
        if self._check_command("apptainer"):
            return "apptainer"

        # Check for Singularity (older name)
        if self._check_command("singularity"):
            return "singularity"

        # Fall back to Docker
        if self._check_command("docker"):
            return "docker"

        raise RuntimeError(
            "No container runtime found. Install Docker or Apptainer.\n"
            "For Apptainer, run: pixi install"
        )

    def execute(self, code: str, session_id: str) -> Dict[str, Any]:
        """
        Execute Python code in isolated container.

        Args:
            code: Python code to execute
            session_id: Unique session identifier

        Returns:
            Dictionary with execution results:
                - status: 'success', 'error', or 'timeout'
                - stdout: Standard output
                - stderr: Standard error
                - plots: List of generated plot filenames
                - output_path: Path to output directory
                - execution_time: Time taken in seconds
        """
        return self.backend.execute(code, session_id)


# Backward compatibility: alias to old name
DockerSandbox = ContainerSandbox
