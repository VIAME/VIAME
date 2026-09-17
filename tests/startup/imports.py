#!/usr/bin/env python3
"""Assert that registering the plugins does not import any of them.

Registering a python plugin must not mean importing the module it lives in:
the declarations in each package's `__init__` carry the interface, the name
and the description, and the import path is resolved only when something
asks for an instance.

The check is the import log rather than the clock, because a clock reading
says the command was fast and not why. If `torch` appears, some package has
gone back to importing its implementations and the three seconds P8-T10
removed are back, whatever the machine happened to time.

`python -X importtime` is the documented way to get this log; VIAME embeds
the interpreter rather than being started by it, so the equivalent
`PYTHONPROFILEIMPORTTIME=1` is what gets set. Same log, same format.

Two commands, because they ask different amounts of the registry:

* `viame runner --help` registers everything and instantiates nothing. This
  is what every command pays, and nothing heavy may appear in it.

* `viame registry-dump` additionally **constructs** every process, because a
  process's ports and config keys do not exist until it is built. So its
  modules are imported by design, and `cv2` is expected: the config keys of
  `gmm_motion_detector` come from `stereo_algos.GMMForegroundObjectDetector
  .default_params()`, an OpenCV algorithm, and declaring them is part of
  constructing the process. `torch` and `mmdet` still may not appear -- the
  pytorch processes reach their models from `_configure`, not from
  `__init__`, so building one to read its ports stays cheap.

Usage:
    imports.py --command 'viame runner --help' [--forbid torch cv2 mmdet]
"""

import argparse
import os
import re
import subprocess
import sys

# `import time: self [us] | cumulative | imported package`
LINE = re.compile(r"^import time:\s*\d+\s*\|\s*\d+\s*\|\s*(\S.*)$")


def imported_modules(command):
    """Every module name the run imported, from its importtime log."""
    environment = dict(os.environ, PYTHONPROFILEIMPORTTIME="1")

    result = subprocess.run(command, shell=True, capture_output=True,
                            env=environment)

    if result.returncode != 0:
        raise RuntimeError(
            "'{}' exited {}:\n{}".format(
                command, result.returncode,
                result.stderr.decode("utf-8", "replace")[-2000:]))

    modules = set()

    for line in result.stderr.decode("utf-8", "replace").splitlines():
        match = LINE.match(line)
        if match:
            modules.add(match.group(1).strip())

    if not modules:
        raise RuntimeError(
            "no import log from '{}'; PYTHONPROFILEIMPORTTIME produced "
            "nothing, so this test is not checking anything".format(command))

    return modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True)
    parser.add_argument("--forbid", nargs="+",
                        default=["torch", "cv2", "mmdet"])
    args = parser.parse_args()

    modules = imported_modules(args.command)
    print("{} modules imported".format(len(modules)))

    failures = []

    for name in args.forbid:
        # A top-level import and a submodule of it both count: `torch.nn`
        # without `torch` is not a thing that happens, but saying so here
        # means the failure names the module that pulled it in.
        offenders = sorted(m for m in modules
                           if m == name or m.startswith(name + "."))
        if offenders:
            failures.append(
                "{} was imported: {}".format(name, ", ".join(offenders[:5])))

    if failures:
        print("\n".join(failures), file=sys.stderr)
        print("\nA package is importing its implementations instead of "
              "declaring them; see viame.plugins.discovery.",
              file=sys.stderr)
        return 1

    print("none of {} imported".format(", ".join(args.forbid)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
