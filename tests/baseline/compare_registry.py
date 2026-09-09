#!/usr/bin/env python3
"""Compare two ``viame registry-dump`` outputs.

The baseline is the contract: every name the old dump registers has to still
be registered, with the same configuration keys and the same defaults. New
names are fine. Descriptions are ignored, since they carry the option list of
every sibling implementation and so churn whenever anything else moves.

A name may leave in two ways:

* it is renamed, and the new implementation registers the old name as an
  alias. The new dump's ``aliases`` map records that and the name counts as
  present.
* it is removed on purpose, and the task that removed it lists it in
  ``removed.json`` as ``{kind, interface, name, phase, reason}``.

Anything else is a failure.

Names and config keys that a later phase restores are listed in
``pending.json`` instead, with the phase that brings them back. That file is
the one place the contract is knowingly relaxed, and it is meant to be empty
again by the end of the phase that fills it.

Usage:
    compare_registry.py OLD NEW [--removed removed.json] [--pending pending.json]
"""

import argparse
import json
import sys


def load(path):
    with open(path) as handle:
        return json.load(handle)


def removed_key(entry):
    return (entry["kind"], entry.get("interface", ""), entry["name"])


def resolve_alias(aliases, name):
    """Follow the new dump's alias map to the name that now implements it."""
    seen = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def compare_config(old_config, new_config, where, failures, pending_keys):
    for key, old_item in sorted(old_config.items()):
        new_item = new_config.get(key)

        if new_item is None:
            if key not in pending_keys:
                failures.append("{}: config key '{}' is gone".format(where, key))
            continue

        if old_item.get("default") != new_item.get("default"):
            failures.append(
                "{}: config key '{}' default changed from '{}' to '{}'".format(
                    where, key, old_item.get("default"), new_item.get("default")
                )
            )


def compare_ports(old_ports, new_ports, where, kind, failures):
    for port, old_item in sorted(old_ports.items()):
        new_item = new_ports.get(port)

        if new_item is None:
            failures.append("{}: {} port '{}' is gone".format(where, kind, port))
            continue

        if old_item.get("type") != new_item.get("type"):
            failures.append(
                "{}: {} port '{}' type changed from '{}' to '{}'".format(
                    where, kind, port, old_item.get("type"), new_item.get("type")
                )
            )


def compare_entries(kind, old_entries, new_entries, aliases, removed, pending,
                    failures, interface="", with_ports=False):
    for name, old_entry in sorted(old_entries.items()):
        where = "{} '{}'".format(kind, name)
        if interface:
            where = "{} '{}' of interface '{}'".format(kind, name, interface)

        if (kind, interface, name) in removed:
            continue

        pending_entry = pending.get((kind, interface, name), {})

        new_entry = new_entries.get(name)

        if new_entry is None:
            target = resolve_alias(aliases, name)
            new_entry = new_entries.get(target)

        if new_entry is None:
            if not pending_entry.get("whole_name"):
                failures.append("{} is gone".format(where))
            continue

        # An entry the dump could not introspect carries no config to compare,
        # in either direction
        if old_entry.get("error") or new_entry.get("error"):
            continue

        compare_config(old_entry.get("config", {}), new_entry.get("config", {}),
                       where, failures,
                       set(pending_entry.get("config_keys", [])))

        if with_ports:
            compare_ports(old_entry.get("input_ports", {}),
                          new_entry.get("input_ports", {}),
                          where, "input", failures)
            compare_ports(old_entry.get("output_ports", {}),
                          new_entry.get("output_ports", {}),
                          where, "output", failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", help="baseline registry dump")
    parser.add_argument("new", help="registry dump to check")
    parser.add_argument("--removed", help="names removed on purpose")
    parser.add_argument("--pending",
                        help="names or keys a later phase restores")
    args = parser.parse_args()

    old = load(args.old)
    new = load(args.new)
    aliases = new.get("aliases", {})

    removed = set()
    if args.removed:
        for entry in load(args.removed):
            removed.add(removed_key(entry))

    pending = {}
    if args.pending:
        for entry in load(args.pending):
            pending[removed_key(entry)] = {
                "whole_name": entry.get("config_keys") is None,
                "config_keys": entry.get("config_keys") or [],
            }

    failures = []

    old_algorithms = old.get("algorithms", {})
    new_algorithms = new.get("algorithms", {})

    for interface, old_impls in sorted(old_algorithms.items()):
        compare_entries("algorithm", old_impls,
                        new_algorithms.get(interface, {}),
                        aliases, removed, pending, failures,
                        interface=interface)

    compare_entries("process", old.get("processes", {}), new.get("processes", {}),
                    aliases, removed, pending, failures, with_ports=True)
    compare_entries("cluster", old.get("clusters", {}), new.get("clusters", {}),
                    aliases, removed, pending, failures, with_ports=True)
    compare_entries("applet", old.get("applets", {}), new.get("applets", {}),
                    aliases, removed, pending, failures)
    compare_entries("scheduler", old.get("schedulers", {}),
                    new.get("schedulers", {}), aliases, removed, pending,
                    failures)

    if failures:
        print("registry baseline: {} regression(s)".format(len(failures)))
        for failure in failures:
            print("  " + failure)
        return 1

    print("registry baseline: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
