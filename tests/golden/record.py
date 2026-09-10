#!/usr/bin/env python3
"""Record golden outputs for the implementations a phase is about to replace.

Run it against an install that still has the old implementation:

    source <install>/setup_viame.sh
    python3 tests/golden/record.py vxl

It writes the input fixtures (once, then never again: replay must not depend
on regenerating them), every case in `cases.py`, and a manifest recording what
produced them. Re-recording an existing group needs --force, so that a golden
is never quietly redefined by the code it is supposed to be checking.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases as case_spec           # noqa: E402
import codec_cases                  # noqa: E402
import codec_fixtures               # noqa: E402
import fixtures                     # noqa: E402
import imageio_utils                # noqa: E402
import pipeline_runner              # noqa: E402
import runner                       # noqa: E402

INPUTS_DIR = os.path.join(HERE, "inputs")
CODEC_INPUTS_DIR = os.path.join(INPUTS_DIR, codec_cases.INPUT_SUBDIR)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def file_digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def describe(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest(array),
    }


TEST_DATA_DIR = os.path.join(HERE, "..", "pipelines", "pipelines_test_data")


def write_inputs():
    """Write the fixture images, leaving any that already exist alone."""
    os.makedirs(INPUTS_DIR, exist_ok=True)
    written = []

    images = dict(fixtures.build())

    if not all(_fixture_exists(name) for name in case_spec.PIPELINE_INPUTS):
        images.update(dict(fixtures.pipeline_frames(TEST_DATA_DIR)))

    for name, array in sorted(images.items()):
        if _fixture_exists(name):
            continue

        imageio_utils.save(os.path.join(INPUTS_DIR, name), array)
        written.append(name)

    return written


def _fixture_exists(name):
    return any(os.path.exists(os.path.join(INPUTS_DIR, name + ext))
               for ext in (".png", ".npz", ".npy"))


def input_path(name):
    for ext in (".png", ".npz", ".npy"):
        candidate = os.path.join(INPUTS_DIR, name + ext)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("no fixture named '{}'".format(name))


def source_versions():
    """Record what produced the goldens, so a re-recording is comparable."""
    versions = {}

    for name, path in (("viame", os.path.join(HERE, "..", "..")),
                       ("kwiver", os.environ.get("KWIVER_SOURCE_DIR", ""))):
        if not path or not os.path.isdir(path):
            continue
        try:
            versions[name] = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, OSError):
            pass

    versions["install"] = os.environ.get("VIAME_INSTALL", "")
    return versions


def record_image_filters(group_dir, manifest):
    for impl, variants in sorted(case_spec.IMAGE_FILTERS.items()):
        input_names = case_spec.inputs_for(impl)
        arrays = [imageio_utils.load(input_path(name)) for name in input_names]

        for variant, config in variants:
            outputs = runner.run_image_filter(impl, config, arrays)

            case_dir = os.path.join(group_dir, "image_filter", impl, variant)
            os.makedirs(case_dir, exist_ok=True)

            files = {}
            for name, output in zip(input_names, outputs):
                written = imageio_utils.save(os.path.join(case_dir, name), output)
                files[name] = {
                    "file": os.path.relpath(written, group_dir),
                    **describe(output),
                }

            manifest["cases"].append({
                "kind": "image_filter",
                "impl": impl,
                "variant": variant,
                "config": config,
                "inputs": list(input_names),
                "outputs": files,
                "unstable": {
                    name: reason
                    for name in input_names
                    for reason in [case_spec.unstable_reason(impl, variant, name)]
                    if reason
                },
            })
            print("  image_filter {} {} ({} inputs)".format(
                impl, variant, len(outputs)))


def record_image_io(group_dir, manifest):
    for impl, variants in sorted(case_spec.IMAGE_IO.items()):
        input_names = case_spec.STILLS
        paths = [input_path(name) for name in input_names]

        for variant, config in variants:
            outputs = runner.run_image_io_load(impl, config, paths)

            case_dir = os.path.join(group_dir, "image_io", impl, variant)
            os.makedirs(case_dir, exist_ok=True)

            files = {}
            for name, output in zip(input_names, outputs):
                written = imageio_utils.save(os.path.join(case_dir, name), output)
                files[name] = {
                    "file": os.path.relpath(written, group_dir),
                    **describe(output),
                }

            manifest["cases"].append({
                "kind": "image_io",
                "impl": impl,
                "variant": variant,
                "config": config,
                "inputs": list(input_names),
                "outputs": files,
                "unstable": {
                    name: reason
                    for name in input_names
                    for reason in [case_spec.unstable_reason(impl, variant, name)]
                    if reason
                },
            })
            print("  image_io {} {} ({} inputs)".format(impl, variant, len(outputs)))


def record_pipelines(group_dir, manifest):
    for pipeline in case_spec.PIPELINES:
        outputs = pipeline_runner.run(pipeline)

        case_dir = os.path.join(group_dir, "pipeline", pipeline)
        os.makedirs(case_dir, exist_ok=True)

        files = {}
        for name, output in sorted(outputs.items()):
            stem = os.path.splitext(name)[0]
            written = imageio_utils.save(os.path.join(case_dir, stem), output)
            files[name] = {
                "file": os.path.relpath(written, group_dir),
                **describe(output),
            }

        manifest["cases"].append({
            "kind": "pipeline",
            "impl": pipeline,
            "variant": "default",
            "config": {},
            "inputs": list(case_spec.PIPELINE_INPUTS),
            "outputs": files,
            "unstable": {
                name: reason
                for name in files
                for reason in [case_spec.unstable_reason(pipeline, "default")]
                if reason
            },
        })
        print("  pipeline {} ({} frames)".format(pipeline, len(files)))


def write_codec_inputs():
    """Write the encoded containers, leaving any that already exist alone."""
    existing = {name for name, path in codec_fixtures.paths(CODEC_INPUTS_DIR)
                if os.path.exists(path)}

    if len(existing) == len(codec_fixtures.paths(CODEC_INPUTS_DIR)):
        return []

    if existing:
        raise RuntimeError(
            "inputs/{} is half written: {} present. Delete the directory and "
            "record again rather than mixing two generations of fixture."
            .format(codec_cases.INPUT_SUBDIR, ", ".join(sorted(existing))))

    return [name for name, _ in codec_fixtures.build(CODEC_INPUTS_DIR)]


def codec_input_path(name):
    for candidate, path in codec_fixtures.paths(CODEC_INPUTS_DIR):
        if candidate == name:
            return path

    raise FileNotFoundError("no codec fixture named '{}'".format(name))


def record_codec_decode(group_dir, manifest):
    for impl, variants in sorted(codec_cases.IMAGE_IO.items()):
        containers = codec_cases.CONTAINERS
        paths = [codec_input_path(name) for name in containers]

        for variant, config in variants:
            outputs = runner.run_image_io_load(impl, config, paths)

            case_dir = os.path.join(group_dir, "decode", impl, variant)
            os.makedirs(case_dir, exist_ok=True)

            files = {}
            for name, output in zip(containers, outputs):
                written = imageio_utils.save(os.path.join(case_dir, name), output)
                files[name] = {
                    "file": os.path.relpath(written, group_dir),
                    **describe(output),
                }

            manifest["cases"].append({
                "kind": "decode",
                "impl": impl,
                "variant": variant,
                "config": config,
                "inputs": list(containers),
                "outputs": files,
                "unstable": {},
            })
            print("  decode {} {} ({} containers)".format(
                impl, variant, len(outputs)))


def record_codec_round_trip(group_dir, manifest):
    sources = codec_cases.WRITE_SOURCES
    arrays = [imageio_utils.load(input_path(name)) for name in sources]

    for impl, variants in sorted(codec_cases.IMAGE_IO.items()):
        for variant, config in variants:
            for extension in codec_cases.WRITE_EXTENSIONS:
                with tempfile.TemporaryDirectory() as work_dir:
                    outputs = runner.run_image_io_save_load(
                        impl, config, arrays, extension, work_dir)

                tag = variant + extension.replace(".", "_")
                case_dir = os.path.join(group_dir, "round_trip", impl, tag)
                os.makedirs(case_dir, exist_ok=True)

                files = {}
                for name, output in zip(sources, outputs):
                    written = imageio_utils.save(
                        os.path.join(case_dir, name), output)
                    files[name] = {
                        "file": os.path.relpath(written, group_dir),
                        **describe(output),
                    }

                manifest["cases"].append({
                    "kind": "round_trip",
                    "impl": impl,
                    "variant": tag,
                    "config": config,
                    "extension": extension,
                    "inputs": list(sources),
                    "outputs": files,
                    "unstable": {},
                })
                print("  round_trip {} {} ({} sources)".format(
                    impl, tag, len(outputs)))


def record_codecs(group_dir, manifest):
    # The containers are the fixture here, and they are bytes rather than
    # arrays: digest the files, so that a re-recording against a different
    # generation of them is visible in the diff.
    manifest["containers"] = {
        name: {
            "file": os.path.relpath(path, INPUTS_DIR),
            "bytes": os.path.getsize(path),
            "sha256": file_digest(path),
        }
        for name, path in codec_fixtures.paths(CODEC_INPUTS_DIR)
    }

    record_codec_decode(group_dir, manifest)
    record_codec_round_trip(group_dir, manifest)


def record_vxl(group_dir, manifest):
    record_image_filters(group_dir, manifest)
    record_image_io(group_dir, manifest)
    record_pipelines(group_dir, manifest)


GROUPS = {
    "vxl": record_vxl,
    "codecs": record_codecs,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=sorted(GROUPS),
                        help="recording group")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing recording")
    args = parser.parse_args()

    group_dir = os.path.join(HERE, args.group)
    manifest_path = os.path.join(group_dir, "manifest.json")

    if os.path.exists(manifest_path) and not args.force:
        print("{} already recorded; pass --force to re-record".format(args.group))
        return 1

    runner.load_modules()

    written = write_inputs()
    if written:
        print("wrote fixtures: {}".format(", ".join(written)))

    if args.group == "codecs":
        written = write_codec_inputs()
        if written:
            print("wrote codec containers: {}".format(", ".join(written)))

    os.makedirs(group_dir, exist_ok=True)

    manifest = {
        "group": args.group,
        "recorded": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": source_versions(),
        "fixtures": {
            name: describe(imageio_utils.load(input_path(name)))
            for name in sorted(set(case_spec.STILLS) | set(case_spec.SEQUENCE)
                               | set(case_spec.MASKS)
                               | set(case_spec.PIPELINE_INPUTS))
        },
        "cases": [],
    }

    GROUPS[args.group](group_dir, manifest)

    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("recorded {} cases into {}".format(len(manifest["cases"]), group_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
