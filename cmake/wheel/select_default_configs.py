#!/usr/bin/env python3
"""Emit a wheel contents fragment for the pipelines and training
configs a wheel can run.

Run by the `wheel` target before `build_wheel.py`, whose `--contents` takes
more than one file so a generated selection can be layered over the
hand-written list.

**Nothing from `configs/add-ons/` is ever selected.** Add-on packs are model
distributions -- `DEFAULT-FISH`, `GFIT`, `SAM3` and the rest, 8.6 GB of
weights across the ones this install carries -- and they are fetched at
runtime, not shipped. This script only ever looks at `configs/pipelines`,
and the exclusion is asserted rather than assumed: see `_refuse_add_ons`.

**What is selected.** Every pipeline the build installs whose models are
small enough to ship with it -- including those needing none. A pipeline's
model footprint is the total size of every `models/` path it or anything it
`include`s names, and the cut is `--max-model-bytes`, 10 MB by default.

That threshold is not arbitrary: the footprints are bimodal. Of 147 runnable
pipelines, 82 need no model, one needs 1.3 MB, eight need 10-100 MB sharing
215 MB between them, and 56 need over 100 MB each. There is nothing in
between to agonise over.

Models under the cut are packed alongside their pipelines, so a shipped
config never points at a file that is not there. Larger ones ship as add-on
packs, fetched at runtime.

The point is a wheel whose pipelines all run on what it ships. A pipeline
whose `relativepath weight = models/gfit_groups_rf_detr_1728.pth` points at
nothing would be worse than absent: it looks installed and fails at
configure time.

`common_*.pipe` are includes rather than entry points, and are carried when
something selected includes them -- the closure is what is emitted.
"""

import argparse
import re
import sys
from pathlib import Path


PIPELINE = re.compile(r"relativepath\s+[\w:]+\s*=\s*(\S+\.pipe)\b")

INCLUDE = re.compile(r"^\s*include\s+(\S+)", re.M)

# A model reference, however the config spells it.
MODEL = re.compile(r"(?:relativepath\s+\w+\s*=\s*|:\S*model\S*\s+|=\s*)(models/\S+)")

# Looser: anything that looks like weights at all. Used to check the above,
# not to select.
WEIGHTS = re.compile(
    r"\.(?:pth|pt|zip|onnx|weights|caffemodel|safetensors|engine|cfg)\b"
    r"|relativepath\s+(?:deployed|weight|net_config|model)", re.I)

# What a config *writes* is not something it needs. `common_train_detector.conf`
# says `output_file = trained_model.zip`, and `.zip` reads as weights to the
# pattern above -- so every config that includes it, which is every tracker
# and detector trainer, was classed as naming a model in an unknown spelling
# and left out of 0.23.4.
OUTPUT = re.compile(r"^\s*:?[\w:.\-]*output[\w:.\-]*\s*(?:=|\s)", re.I)


def _needs(text):
    """The text with the lines that only say where output goes removed."""
    return "\n".join(line for line in text.splitlines()
                     if not OUTPUT.match(line))


def _refuse_add_ons(path):
    """Add-on content must never reach the wheel."""
    if "add-ons" in path.parts or "add_ons" in path.parts:
        raise SystemExit(
            f"select_default_configs: refusing {path} -- it is add-on "
            f"content, which is fetched at runtime and never shipped")


def closure(entry, root, seen=None):
    """`entry` and everything it includes, transitively."""
    seen = seen if seen is not None else set()
    if entry in seen:
        return seen
    if not entry.is_file():
        raise ValueError(f"missing configuration dependency: {entry}")
    _refuse_add_ons(entry)
    seen.add(entry)
    text = "\n".join(line.split("#", 1)[0] for line in
                     entry.read_text(errors="replace").splitlines())
    if re.search(r"^\s*TODO\b", text, re.M):
        raise ValueError(f"unfinished configuration: {entry}")
    for name in INCLUDE.findall(text) + PIPELINE.findall(text):
        name = name.strip('"\'')
        name = name.replace("$ENV{VIAME_INSTALL}/configs/pipelines/", "")
        closure(root / name, root, seen)
    return seen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True, help="the install prefix")
    p.add_argument("--output", required=True, help="the contents fragment to write")
    p.add_argument("--destination", default="{data}/configs/pipelines/",
                   help="where the configs land inside the wheel. The "
                        "default is the env prefix, which is where `viame` "
                        "looks: `<exe>/../configs`.")
    p.add_argument("--manifest",
                   help="CMake install_manifest.txt. Without it the prefix is "
                        "read as-is, which includes pipelines earlier builds "
                        "left there for features this one has off.")
    p.add_argument("--max-model-bytes", type=int, default=10 * 1024 * 1024,
                   help="ship a pipeline whose models total no more than this; "
                        "larger ones come from add-on packs")
    args = p.parse_args(argv)

    root = Path(args.prefix) / "configs" / "pipelines"
    if not root.is_dir():
        raise SystemExit(f"select_default_configs: {root} is not a directory")

    # "Enabled in this build" is what the manifest says, not what the prefix
    # holds: `make install` only ever adds, so a prefix carries pipelines for
    # options this build has off. Selecting those ships configs the build did
    # not produce -- and, worse, their models, since a model can be present
    # from a downloaded pack while the pipeline that uses it is not built.
    installed = None
    if args.manifest and Path(args.manifest).is_file():
        installed = {str(Path(l.strip()).resolve())
                     for l in Path(args.manifest).read_text().splitlines()
                     if l.strip()}

    def built(path):
        return installed is None or str(path.resolve()) in installed

    # `.conf` as well as `.pipe`: `viame train` reads the training configs,
    # and the same model rule applies to both.
    entries = sorted(list(root.glob("*.pipe")) + list(root.glob("*.conf")))

    selected, models, needs_model, mismatched = set(), set(), 0, []
    training = 0
    # Every entry that is left out, with why. A count alone is how twenty
    # training configs left a release without anyone being told which.
    left_out = []
    for entry in entries:
        if entry.name.startswith("common_") or not built(entry):
            continue        # an include, or not a pipeline this build makes
        try:
            chain = closure(entry, root)
        except ValueError as exc:
            reason = str(exc).replace(str(root) + "/", "")
            left_out.append((entry.name, reason))
            continue
        if any(not built(path) for path in chain):
            left_out.append((entry.name, "includes a file this build did not install"))
            continue
        # A training config ships, with everything it includes. The rule
        # below -- leave out what names a model the build did not install --
        # is about pipelines that *run* a pretrained model, where a missing
        # one means the pipeline cannot start. A trainer's `models/` path is
        # a seed: RF-DETR's configs set `seed_model_url_fallback` and fetch
        # it, the default trainer downloads what it selects, and the rest say
        # which file they want when it is absent. Leaving the config out
        # instead means `viame train` cannot be asked for it at all, even by
        # someone who has the seed.
        if entry.suffix == ".conf" and entry.name.startswith("train_"):
            selected |= chain
            training += 1
            continue
        text = _needs("\n".join(
            line.split("#", 1)[0] for c in chain
            for line in c.read_text(errors="replace").splitlines()))

        refs = {root / r for r in MODEL.findall(text)}
        if refs:
            # A model the build did not install is an add-on that was not
            # downloaded. Its size is unknowable, so it is not ours to ship.
            absent = sorted(f.name for f in refs
                            if not f.is_file() or not built(f))
            if absent:
                needs_model += 1
                left_out.append((entry.name, "needs " + ", ".join(absent)))
                continue
            if sum(f.stat().st_size for f in refs) > args.max_model_bytes:
                needs_model += 1
                left_out.append((entry.name, "its models are over the size cut"))
                continue
            models |= refs
            selected |= chain
            continue
        if WEIGHTS.search(text):
            # The loose pattern disagreeing with the strict one means a model
            # is referenced in a spelling `MODEL` does not know. Report it
            # rather than ship a pipeline that cannot run.
            mismatched.append(entry.name)
            left_out.append((entry.name,
                             "mentions weights in a spelling not recognised"))
            continue
        selected |= chain

    selected = {p for p in selected if built(p)}

    if mismatched:
        print(f"  note: {len(mismatched)} pipeline(s) name no `models/` path but "
              f"do mention weights; left out: {mismatched[:4]}", file=sys.stderr)

    if not selected:
        raise SystemExit("select_default_configs: selected nothing")

    lines = [
        "# GENERATED by select_default_configs.py -- do not edit.",
        "#",
        "# The pipelines whose models are small enough to ship, the",
        "# `common_*.pipe` they include, and those models. Nothing here comes",
        "# from `configs/add-ons/`: add-on packs are model distributions,",
        "# fetched at runtime and never shipped.",
        "",
    ]
    for path in sorted(selected):
        rel = path.relative_to(Path(args.prefix)).as_posix()
        lines.append(f"include {rel} -> {args.destination}{path.relative_to(root).as_posix()}")
    for path in sorted(models):
        rel = path.relative_to(Path(args.prefix)).as_posix()
        lines.append(f"include {rel} -> {args.destination}models/{path.name}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    if not out.is_file() or out.read_text() != text:
        out.write_text(text)

    size = sum(p.stat().st_size for p in selected)
    msize = sum(p.stat().st_size for p in models)
    extra = f" and {len(models)} model(s), {msize / 1024:.0f} KB" if models else ""
    print(f"  default configs: {len(selected)} files, {size / 1024:.0f} KB{extra}, "
          f"{training} training config(s) among them "
          f"({needs_model} pipelines left out as needing add-on models)")
    for name, reason in sorted(left_out):
        print(f"    left out: {name} -- {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
