# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""DIVE's interactive service command line must keep starting the service.

DIVE's desktop client runs `python -s -m viame.core.interactive_service`
(`client/platform/desktop/backend/native/interactive.ts`). The module is
`viame.segmentation.interactive_service` since P2-T05, and `viame.core` was
emptied by P2-T07, so the old path is a forwarder. `--help` is the cheapest
thing that proves the forwarder reaches the real module's argument parser:
it loads no model and needs no GPU.
"""

import importlib
import subprocess
import sys


def _help(module):
    return subprocess.run(
        [sys.executable, "-s", "-m", module, "--help"],
        capture_output=True, text=True, timeout=120)


def test_old_path_reaches_the_service():
    old = _help("viame.core.interactive_service")
    new = _help("viame.segmentation.interactive_service")

    assert old.returncode == 0, old.stderr
    assert new.returncode == 0, new.stderr
    assert "--segmentation-config" in old.stdout
    assert old.stdout == new.stdout


def test_forwarder_package_registers_nothing():
    package = importlib.import_module("viame.core")

    assert package.__vital_algorithm_declarations__ == []
    assert package.__sprokit_process_declarations__ == []
