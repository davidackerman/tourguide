"""
Apptainer Sandbox for Secure Code Execution.

Executes Python code in isolated Apptainer containers with resource limits
and security restrictions.
"""

import os
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List


class ApptainerBackend:
    """Manages secure execution of Python code in Apptainer containers."""

    IMAGE_NAME = "tourguide-analysis.sif"

    def __init__(
        self,
        db_path: str,
        timeout: int = 60,
        memory_limit: str = "512m",
        cpu_quota: int = 50000
    ):
        """
        Initialize Apptainer backend.

        Args:
            db_path: Path to organelle database
            timeout: Execution timeout in seconds (default: 60)
            memory_limit: Container memory limit (default: 512m)
            cpu_quota: CPU quota (not directly used in Apptainer, LSF handles this)
        """
        self.db_path = Path(db_path).resolve()
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.results_base_dir = Path("analysis_results")
        self.containers_dir = Path("containers")

        # Ensure directories exist
        self.results_base_dir.mkdir(exist_ok=True)
        self.containers_dir.mkdir(exist_ok=True)

        # Get image path from env or use default
        image_path = os.environ.get("APPTAINER_IMAGE_PATH")
        if image_path:
            self.image_path = Path(image_path).resolve()
        else:
            self.image_path = self.containers_dir / self.IMAGE_NAME

        # Check if Apptainer is available
        if not self._check_apptainer():
            raise RuntimeError("Apptainer is not available. Install Apptainer (run 'pixi install').")

        # Build image if needed
        self._ensure_image()

        print(f"[APPTAINER_BACKEND] Initialized (timeout={timeout}s, mem={memory_limit})")
        print(f"[APPTAINER_BACKEND] Using image: {self.image_path}")

    def _check_apptainer(self) -> bool:
        """
        Check if Apptainer is available.

        Returns:
            True if Apptainer is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["apptainer", "version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Try singularity as fallback
            try:
                result = subprocess.run(
                    ["singularity", "version"],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

    def _get_command(self) -> str:
        """
        Get the container command (apptainer or singularity).

        Returns:
            Command name
        """
        # Try apptainer first
        try:
            result = subprocess.run(
                ["apptainer", "version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return "apptainer"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fall back to singularity
        try:
            result = subprocess.run(
                ["singularity", "version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return "singularity"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        raise RuntimeError("Neither apptainer nor singularity command found")

    def _ensure_image(self):
        """Build Apptainer image if it doesn't exist."""
        if self.image_path.exists():
            print(f"[APPTAINER_BACKEND] Using existing image: {self.image_path}")
            return

        print(f"[APPTAINER_BACKEND] Building image: {self.image_path}")
        self._build_image()

    def _build_image(self):
        """Build the analysis Apptainer image."""
        definition_file = Path(__file__).parent / "analysis_container.def"

        if not definition_file.exists():
            raise RuntimeError(f"Definition file not found: {definition_file}")

        cmd = self._get_command()

        try:
            # Build with --fakeroot (user-level build, no sudo needed)
            subprocess.run(
                [cmd, "build", "--fakeroot", str(self.image_path), str(definition_file)],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"[APPTAINER_BACKEND] Successfully built image: {self.image_path}")
        except subprocess.CalledProcessError as e:
            print(f"[APPTAINER_BACKEND] Failed to build image:")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise RuntimeError(f"Failed to build Apptainer image: {e.stderr}")

    def execute(self, code: str, session_id: str) -> Dict[str, Any]:
        """
        Execute Python code in isolated Apptainer container.

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
            print(f"[APPTAINER_BACKEND] Execution failed: {e}")
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
        Run Apptainer container with code.

        Args:
            session_dir: Session directory containing code and output

        Returns:
            Dictionary with status, stdout, stderr
        """
        cmd = self._get_command()

        # Build apptainer exec command
        apptainer_cmd = [
            cmd, "exec",
            "--contain",                            # Minimal isolation
            "--no-home",                            # Don't auto-mount home
            "--bind", f"{self.db_path.parent}:/data:ro",  # Database (read-only)
            "--bind", f"{session_dir}:/code:ro",    # Code (read-only)
            "--bind", f"{session_dir}:/output:rw",  # Output (read-write)
            "--pwd", "/code",                       # Working directory
        ]

        # Add memory limit if supported (optional, LSF may handle this)
        # Note: --memory flag might not be available in all Apptainer versions
        # We'll skip it for now and rely on LSF scheduler for cluster

        # Add image and command
        apptainer_cmd.extend([
            str(self.image_path),
            "python", "/code/analysis.py"
        ])

        print(f"[APPTAINER_BACKEND] Running container with timeout={self.timeout}s")

        try:
            # Run with timeout
            result = subprocess.run(
                apptainer_cmd,
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
            print(f"[APPTAINER_BACKEND] Execution timed out after {self.timeout}s")

            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Execution exceeded timeout of {self.timeout} seconds"
            }

        except Exception as e:
            print(f"[APPTAINER_BACKEND] Container execution error: {e}")
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

        print(f"[APPTAINER_BACKEND] Found {len(plot_files)} plot file(s): {plot_files}")
        return sorted(plot_files)
