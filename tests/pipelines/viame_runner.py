import subprocess
import os

from pathlib import Path
from functools import cached_property

class ViameRunner:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir
        self._viame_env = None

    @property
    def viame_source(self):
        """Get the VIAME source directory."""
        return Path(__file__).resolve().parent.parent.parent

    @cached_property
    def viame_install(self):
        """Get the VIAME install directory."""
        if 'VIAME_INSTALL' in os.environ:
            install_path = Path(os.environ['VIAME_INSTALL'])
            if install_path.exists():
                return install_path

        source = self.viame_source
        candidates = [
            source.parent / "build" / "install",
            source / "build" / "install",
        ]

        for candidate in candidates:
            if (candidate / "setup_viame.sh").exists():
                return candidate

        raise ValueError("VIAME install directory not found. Set VIAME_INSTALL environment variable.")

    def _get_sourced_env(self):
        if self._viame_env:
            return self._viame_env

        setup_script = self.viame_install / ("setup_viame.sh" if os.name != 'nt' else "setup_viame.bat")

        if not setup_script.exists():
            raise FileNotFoundError(f"Setup script not found : {setup_script}")

        command = f"source {setup_script} && env" if os.name != 'nt' else f"call {setup_script} && set"
        result = subprocess.check_output(command, shell=True, executable="/bin/bash" if os.name != 'nt' else None, text=True)

        new_env = os.environ.copy()
        for line in result.splitlines():
            if '=' in line:
                key, _, value = line.partition('=')
                new_env[key] = value

        self._viame_env = new_env
        return new_env

    def run(self, pipeline_path: Path | str, workdir, overrides=None):
        env = self._get_sourced_env()
        if isinstance(pipeline_path, str):
            pipeline_path = Path(self.viame_install, "configs", pipeline_path)

        cmd = ["kwiver", "runner", str(pipeline_path)]
        if overrides:
            for k, v in overrides.items():
                cmd += ["-s", f"{k}={v}"]

        return subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, env=env)