#!/usr/bin/env python3
"""Time the `viame` commands that must not pay for the plugin system.

`viame --version` and `viame runner --help` answer from the applet table
alone. `viame registry-dump` genuinely walks every plugin, but walking them
must not mean importing them: it reads names and descriptions out of the
declarations in each python package's `__init__`, and imports an
implementation module only when something asks for an instance.

Before P8-T10 every python package imported all of its implementations at
registration time, so all three commands paid for torch, opencv and mmdet
whether or not they were going to use them.

What is measured is the **best** of several runs, not the mean: the budget is
a claim about the work the command does, and a fair machine will hit the
floor. A loaded machine adds noise upwards only, and a mean would turn other
tests running beside this one into a failure here.

Usage:
    budget.py [--runs N] [--json OUT] NAME=BUDGET:COMMAND ...
"""

import argparse
import json
import subprocess
import sys
import time


def run_once(command):
    """Wall seconds for one run. Raises if the command fails."""
    start = time.perf_counter()
    result = subprocess.run(command, shell=True, capture_output=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            "'{}' exited {}:\n{}".format(
                command, result.returncode,
                result.stderr.decode("utf-8", "replace")[-2000:]))

    return elapsed


def measure(command, runs):
    """The best of `runs` timings, after one unmeasured warm-up."""
    run_once(command)
    return min(run_once(command) for _ in range(runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3,
                        help="timed runs per command (default 3)")
    parser.add_argument("--json", help="write the measurements here")
    parser.add_argument("cases", nargs="+", metavar="NAME=BUDGET:COMMAND")
    args = parser.parse_args()

    measurements = {}
    failures = []

    for case in args.cases:
        name, _, rest = case.partition("=")
        budget, _, command = rest.partition(":")
        budget = float(budget)

        try:
            best = measure(command, args.runs)
        except RuntimeError as error:
            failures.append(str(error))
            continue

        measurements[name] = {"seconds": round(best, 3), "budget": budget}

        print("{:<24} {:6.3f} s   budget {:.1f} s   {}".format(
            name, best, budget, "ok" if best <= budget else "OVER"))

        if best > budget:
            failures.append(
                "{} took {:.3f} s, over its {:.1f} s budget".format(
                    name, best, budget))

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(measurements, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
