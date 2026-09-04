import os
import signal
import subprocess

from pathlib import Path
from functools import cached_property

from viame_env import find_viame_install, get_sourced_env

PIPELINE_TIMEOUT = 900


class ViameRunner:
    def __init__(self):
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

    def run(self, pipeline_path: Path | str, workdir, overrides=None,
            timeout: int = PIPELINE_TIMEOUT):
        env = self._get_sourced_env()
        if isinstance(pipeline_path, str):
            pipeline_path = Path(self.viame_install, "configs", pipeline_path)

        cmd = ["kwiver", "runner", str(pipeline_path)]
        if overrides:
            for k, v in overrides.items():
                cmd += ["-s", f"{k}={v}"]

        # Own process group so a timeout can kill everything the pipeline
        # spawned; an orphaned trainer otherwise holds GPU memory across tests.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=workdir, env=env, start_new_session=True)

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_group(proc)
            # A deadlocked pipeline would otherwise consume the whole ctest budget.
            raise AssertionError(
                f"{pipeline_path.name} did not finish within {timeout}s") from None

        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _terminate_group(proc, grace: float = 10.0):
    """
    Kill a process and every descendant it started.

    The process is a group leader courtesy of start_new_session, so its pid
    doubles as the group id. SIGTERM first, so a pipeline gets the chance to
    close its outputs, then SIGKILL for anything still standing.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.communicate(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
