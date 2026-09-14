#!/usr/bin/env python3
"""Check `lite-file-map.tsv` against the tables in `lite-library-layout.md`.

The map was generated from rules; the layout document is what the plan
actually says. Where they disagree the document wins, and the whole point of
this script is that the disagreements are found once rather than one
capability at a time by noticing that a file landed somewhere odd.

It reads the `## 3. Mapping` section, expands `{a,b}` groups, drops the
parenthetical asides, and for every name it can tie to a real file under
`plugins/` reports where the document puts it and where the map does.
"""

import fnmatch
import os
import re
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DOC = os.path.join(ROOT, "design", "lite-library-layout.md")
MAP = os.path.join(ROOT, "design", "lite-file-map.tsv")

# Sections of the document, and the plugin directory each one is about.
SECTION_PLUGIN = {
    "### `plugins/core`": "plugins/core",
    "### `plugins/opencv`": "plugins/opencv",
    "### `plugins/pytorch`": "plugins/pytorch",
    "### Remaining plugins": None,
}


def expand(name):
    """`read_{a,b}_x` -> [`read_a_x`, `read_b_x`]."""
    match = re.search(r"\{([^}]*)\}", name)
    if not match:
        return [name]
    out = []
    for piece in match.group(1).split(","):
        out.extend(expand(name[:match.start()] + piece + name[match.end():]))
    return out


def doc_assignments():
    """[(destination directory, stem, plugin directory or None)]."""
    rows = []
    plugin = None
    started = False

    for line in open(DOC):
        line = line.rstrip("\n")
        if line.startswith("## 3. Mapping"):
            started = True
            continue
        if not started:
            continue
        if line.startswith("## ") and not line.startswith("## 3."):
            break
        if line in SECTION_PLUGIN:
            plugin = SECTION_PLUGIN[line]
            continue
        if not line.startswith("|") or line.startswith("|---"):
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2 or cells[0] in ("Destination", "Plugin", "Files"):
            continue

        left, right = cells

        if plugin is None:
            # The "Remaining plugins" table is the other way round: the left
            # cell names the plugin and the right cell holds
            # `destination/` (`name`, `name`) groups.
            source = re.sub(r"[`/]", "", left)
            # `object_detectors/python` (`onnx_predictor`, ...),
            # `classifiers/` (`refine_detections_svm`), `file_io/database/`
            # (`*_db` algos). The pattern used to insist on a trailing slash
            # inside the backticks, so the `/python` rows -- all of onnx and
            # colmap -- matched nothing and were never checked.
            for destination, names in re.findall(
                    r"`([a-z_0-9]+(?:/[a-z_0-9]*)*)/?`\s*\(([^)]*)\)",
                    right):
                for name in re.findall(r"`([^`]+)`", names):
                    for stem in expand(name):
                        rows.append((destination.split("/")[0],
                                     os.path.basename(stem),
                                     "plugins/" + source))
            continue

        # `object_detectors/python` is `object_detectors`: P2-T01 puts the
        # python beside the C++ rather than in a `python/` subdirectory.
        destination = left.strip("`").split("/")[0]
        if destination == "wheels (P9)":
            continue
        if destination == "deleted":
            destination = "DELETE"
        # `\`utilities.py\` -> \`base.py\`` is one assignment, not two: the
        # second name is what the file is called afterwards.
        right = re.sub(r"->\s*`[^`]+`", "", right)

        for name in re.findall(r"`([^`]+)`", right):
            for stem in expand(name):
                stem = stem.strip()
                if not stem:
                    continue
                # `torchvision/*_feature_extractor` and `detectron2/` name a
                # directory rather than a file. They used to be skipped,
                # which is how the map came to put `torchvision/` in
                # `object_detectors` and keep `detectron2/` the document
                # deletes. They are checked as a prefix now.
                if "*" in stem or stem.endswith("/"):
                    rows.append((destination.split("/")[0],
                                 "PATTERN:" + stem, plugin))
                    continue
                # `onnx_exporters/yolomit_to_onnx` names that file, not every
                # `yolomit_to_onnx` under the plugin: keep the relative path
                rows.append((destination.split("/")[0],
                             stem if "/" in stem else os.path.basename(stem),
                             plugin))
    return rows


def main():
    mapped = {}
    for line in open(MAP):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        source, destination = line.split("\t")
        mapped[source] = destination

    tracked = subprocess.run(["git", "ls-files", "plugins"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    # Also the ones already moved, found by their mapped destination.
    for source in mapped:
        if source not in tracked:
            tracked.append(source)

    by_stem = {}
    for path in tracked:
        stem = os.path.splitext(os.path.basename(path))[0]
        by_stem.setdefault(stem, []).append(path)

    disagree = []
    unknown = []

    # A file the document names under two destinations -- `stereo_algos.py`,
    # which holds the GMM detector as well as the stereo maths -- is placed
    # correctly if it is under either.
    allowed = {}
    for destination, stem, plugin in doc_assignments():
        allowed.setdefault((os.path.splitext(stem)[0], plugin),
                           set()).add(destination)

    # A file the document names outright is placed by that name, whatever a
    # pattern elsewhere would say: `remax_base_trainer.py` is in the
    # `deleted` row by name and would otherwise also match `*_trainer.py`.
    named = set()
    for _, stem, plugin in doc_assignments():
        if not stem.startswith("PATTERN:") and plugin:
            named.add((plugin, os.path.splitext(stem)[0]))

    for destination, stem, plugin in doc_assignments():
        if stem.startswith("PATTERN:"):
            if not plugin:
                continue
            pattern = stem[len("PATTERN:"):]
            for path, actual in mapped.items():
                if not path.startswith(plugin + "/"):
                    continue
                relative = path[len(plugin) + 1:]
                if pattern.endswith("/"):
                    # `detectron2/`: everything under that directory.
                    hit = relative.startswith(pattern)
                else:
                    # `*_trainer.py`, `torchvision/*_feature_extractor`: a
                    # name pattern within one directory. `fnmatch`'s `*`
                    # crosses `/`, so the directories are compared first --
                    # otherwise `*_trainer.py` would claim every file in
                    # every vendored subtree.
                    want_dir, want_name = os.path.split(pattern)
                    have_dir, have_name = os.path.split(relative)
                    if not os.path.splitext(want_name)[1]:
                        have_name = os.path.splitext(have_name)[0]
                    hit = (want_dir == have_dir
                           and fnmatch.fnmatch(have_name, want_name))
                if not hit or actual == "STRUCTURAL":
                    continue
                if (os.path.dirname(path) == plugin and
                        (plugin, os.path.splitext(os.path.basename(path))[0])
                        in named):
                    continue
                actual_dir = (actual.split("/")[1]
                              if actual.startswith("library/") else actual)
                if actual_dir != destination:
                    disagree.append((path, actual_dir, destination))
            continue
        stem = os.path.splitext(stem)[0]
        if "/" in stem and plugin:
            # A name below the plugin directory, matched by that path
            candidates = [p for p in tracked
                          if os.path.splitext(p)[0] == plugin + "/" + stem]
            if not candidates:
                unknown.append((destination, stem, plugin))
                continue
        else:
            candidates = by_stem.get(stem, [])
        if plugin and "/" not in stem:
            # Directly in the plugin directory: a document entry naming
            # `utilities.py` means `plugins/pytorch/utilities.py`, not every
            # `utilities.py` in every vendored subtree below it.
            candidates = [c for c in candidates
                          if os.path.dirname(c) == plugin]
        if not candidates:
            unknown.append((destination, stem, plugin))
            continue
        for path in candidates:
            actual = mapped.get(path, "?")
            if actual == "STRUCTURAL":
                continue
            actual_dir = actual.split("/")[1] if actual.startswith("library/") \
                else actual
            wanted = allowed.get((stem, plugin), {destination})
            if actual_dir not in wanted:
                disagree.append((path, actual_dir, "|".join(sorted(wanted))))

    for path, actual, wanted in sorted(set(disagree)):
        print("%-55s map=%-20s doc=%s" % (path, actual, wanted))
    print("\n%d disagreements, %d names in the document with no file"
          % (len(set(disagree)), len(unknown)))
    if "-v" in sys.argv:
        for destination, stem, plugin in unknown:
            print("   no file for %s (%s -> %s)" % (stem, plugin, destination))
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
