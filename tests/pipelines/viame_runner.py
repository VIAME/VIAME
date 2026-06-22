import subprocess

from pathlib import Path
from functools import cached_property

from viame_env import find_viame_install, get_sourced_env


class ViameRunner:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir
        self._viame_env = None

    @cached_property
    def viame_install(self):
        """Get the VIAME install directory."""
        install = find_viame_install()
        if install is None:
            raise ValueError("VIAME install directory not found. Set VIAME_INSTALL environment variable.")
        return install

    def _get_sourced_env(self):
        if self._viame_env is None:
            self._viame_env = get_sourced_env(self.viame_install)
        return self._viame_env

    def run(self, pipeline_path: Path | str, workdir, overrides=None):
        env = self._get_sourced_env()
        if isinstance(pipeline_path, str):
            pipeline_path = Path(self.viame_install, "configs", pipeline_path)

        cmd = ["kwiver", "runner", str(pipeline_path)]
        if overrides:
            for k, v in overrides.items():
                cmd += ["-s", f"{k}={v}"]

        return subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, env=env)
