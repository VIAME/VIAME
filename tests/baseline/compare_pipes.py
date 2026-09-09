#!/usr/bin/env python3
"""Compare two ``viame pipe-check --all`` outputs.

The baseline records, for every pipeline and configuration file the install
ships, whether it bakes and what each of its processes resolves to. A file may
not start failing, may not lose a process, and may not silently start using a
different implementation. Files whose baseline status is already ``error`` are
held to that: they have to keep failing the same way, so that a fix is noticed
too and the baseline is updated deliberately.

New files are fine. A file that disappears is a failure unless it is listed in
the optional removed-files list.

Usage:
    compare_pipes.py OLD NEW [--removed removed_pipes.json]
"""

import argparse
import json
import sys


def load(path):
    with open(path) as handle:
        return json.load(handle)


def compare_algos(old_algos, new_algos, where, failures):
    for key, old_item in sorted(old_algos.items()):
        new_item = new_algos.get(key)

        if new_item is None:
            failures.append("{}: '{}' is no longer selected".format(where, key))
            continue

        if old_item.get("impl") != new_item.get("impl"):
            failures.append(
                "{}: '{}' changed from '{}' to '{}'".format(
                    where, key, old_item.get("impl"), new_item.get("impl")
                )
            )
        elif old_item.get("resolved") and not new_item.get("resolved"):
            failures.append(
                "{}: '{}' no longer resolves to a registered '{}'".format(
                    where, key, old_item.get("impl")
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", help="baseline pipe check")
    parser.add_argument("new", help="pipe check to compare")
    parser.add_argument("--removed", help="files removed on purpose")
    args = parser.parse_args()

    old = load(args.old)
    new = load(args.new)

    removed = set(load(args.removed)) if args.removed else set()

    failures = []

    for path, old_result in sorted(old.items()):
        if path in removed:
            continue

        new_result = new.get(path)

        if new_result is None:
            failures.append("{} is gone".format(path))
            continue

        if old_result.get("status") != new_result.get("status"):
            failures.append(
                "{}: status changed from {} to {}{}".format(
                    path, old_result.get("status"), new_result.get("status"),
                    ": " + new_result.get("message", "")
                    if new_result.get("status") == "error" else ""
                )
            )
            continue

        old_processes = old_result.get("processes", {})
        new_processes = new_result.get("processes", {})

        for name, old_process in sorted(old_processes.items()):
            new_process = new_processes.get(name)

            if new_process is None:
                failures.append("{}: process '{}' is gone".format(path, name))
                continue

            if old_process.get("type") != new_process.get("type"):
                failures.append(
                    "{}: process '{}' type changed from '{}' to '{}'".format(
                        path, name, old_process.get("type"),
                        new_process.get("type")
                    )
                )

            compare_algos(old_process.get("algos", {}),
                          new_process.get("algos", {}),
                          "{} process '{}'".format(path, name), failures)

        compare_algos(old_result.get("algos", {}), new_result.get("algos", {}),
                      path, failures)

    if failures:
        print("pipe baseline: {} regression(s)".format(len(failures)))
        for failure in failures:
            print("  " + failure)
        return 1

    print("pipe baseline: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
