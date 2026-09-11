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
import calib_cases                  # noqa: E402
import calib_runner                 # noqa: E402
import codec_cases                  # noqa: E402
import codec_fixtures               # noqa: E402
import feature_cases                # noqa: E402
import feature_runner               # noqa: E402
import fixtures                     # noqa: E402
import opencv_cases                 # noqa: E402
import opencv_fixtures              # noqa: E402
import imageio_utils                # noqa: E402
import measurement_cases            # noqa: E402
import measurement_fixtures         # noqa: E402
import measurement_runner           # noqa: E402
import pipeline_runner              # noqa: E402
import refine_cases                 # noqa: E402
import refine_runner                # noqa: E402
import runner                       # noqa: E402
import warp_cases                   # noqa: E402
import warp_runner                  # noqa: E402

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


REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
TEST_DATA_DIR = os.path.join(HERE, "..", "pipelines", "pipelines_test_data")


def write_inputs():
    """Write the fixture images, leaving any that already exist alone."""
    os.makedirs(INPUTS_DIR, exist_ok=True)
    written = []

    images = dict(fixtures.build())
    images.update(opencv_fixtures.build())
    images.update(measurement_fixtures.build())

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


def _write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def record_calibrations(group_dir, manifest):
    case_dir = os.path.join(group_dir, "calibration")
    os.makedirs(case_dir, exist_ok=True)

    files = {}
    for name, source in sorted(calib_cases.CALIBRATIONS.items()):
        loaded = calib_runner.load_calibration(
            os.path.join(REPO_ROOT, source))
        payload = {key: loaded[key].tolist()
                   for key in calib_cases.CALIBRATION_KEYS}
        written = _write_json(os.path.join(case_dir, name + ".json"), payload)
        files[name] = {
            "file": os.path.relpath(written, group_dir),
            "source": source,
            "sha256": file_digest(written),
        }
        print("  calibration {} ({} keys)".format(name, len(payload)))

    manifest["cases"].append({
        "kind": "calibration",
        "impl": "read_stereo_rig",
        "variant": "defaults",
        "config": {},
        "inputs": sorted(calib_cases.CALIBRATIONS),
        "outputs": files,
        "unstable": {},
    })


def record_documents(group_dir, manifest):
    case_dir = os.path.join(group_dir, "nodes")
    os.makedirs(case_dir, exist_ok=True)

    files = {}
    for name, source in sorted(calib_cases.DOCUMENTS.items()):
        payload = calib_runner.dump_document_reference(
            os.path.join(REPO_ROOT, source))
        written = _write_json(os.path.join(case_dir, name + ".json"), payload)
        files[name] = {
            "file": os.path.relpath(written, group_dir),
            "source": source,
            "sha256": file_digest(written),
        }
        print("  nodes {} ({} top level nodes)".format(name, len(payload)))

    manifest["cases"].append({
        "kind": "nodes",
        "impl": "FileStorage",
        "variant": "defaults",
        "config": {},
        "inputs": sorted(calib_cases.DOCUMENTS),
        "outputs": files,
        "unstable": {},
    })


def _record_array_case(group_dir, manifest, kind, impl, variant, config,
                       input_names, outputs, spec, refuses=None):
    """One case whose outputs are one array per input."""
    case_dir = os.path.join(group_dir, kind, impl, variant)
    os.makedirs(case_dir, exist_ok=True)

    files = {}
    for name, output in zip(input_names, outputs):
        written = imageio_utils.save(os.path.join(case_dir, name), output)
        files[name] = {
            "file": os.path.relpath(written, group_dir),
            **describe(output),
        }

    manifest["cases"].append({
        "kind": kind,
        "impl": impl,
        "variant": variant,
        "config": config,
        "inputs": list(input_names),
        "outputs": files,
        "unstable": {
            name: reason
            for name in input_names
            for reason in [spec.unstable_reason(impl, variant, name)]
            if reason
        },
        "refuses": refuses or {},
    })
    print("  {} {} {} ({} inputs{})".format(
        kind, impl, variant, len(outputs),
        ", {} refused".format(len(refuses)) if refuses else ""))


def record_opencv_filters(group_dir, manifest):
    groups = (
        (opencv_cases.IMAGE_FILTERS, None),
        (opencv_cases.BAYER_FILTERS, opencv_cases.BAYER),
        (opencv_cases.TEMPORAL_FILTERS, opencv_cases.SEQUENCE),
    )

    for table, fixed_inputs in groups:
        for impl, variants in sorted(table.items()):
            for variant, config in variants:
                input_names = (fixed_inputs or
                               opencv_cases.filter_inputs(impl, variant))
                arrays = [imageio_utils.load(input_path(name))
                          for name in input_names]

                outputs = runner.run_image_filter(impl, config, arrays)
                _record_array_case(group_dir, manifest, "image_filter", impl,
                                   variant, config, input_names, outputs,
                                   opencv_cases,
                                   refuses=opencv_cases.refusals(impl, variant))


def record_opencv_splits(group_dir, manifest):
    input_names = opencv_cases.STILLS
    arrays = [imageio_utils.load(input_path(name)) for name in input_names]

    for impl, variants in sorted(opencv_cases.SPLIT_IMAGES.items()):
        for variant, config in variants:
            split = runner.run_split_image(impl, config, arrays)

            # One input becomes several images, so the recorded names carry
            # the piece index.
            names, outputs = [], []
            for name, pieces in zip(input_names, split):
                for index, piece in enumerate(pieces):
                    names.append("{}_{}".format(name, index))
                    outputs.append(piece)

            _record_array_case(group_dir, manifest, "split_image", impl,
                               variant, config, names, outputs, opencv_cases)


def record_opencv_motion(group_dir, manifest):
    input_names = opencv_cases.SEQUENCE
    arrays = [imageio_utils.load(input_path(name)) for name in input_names]

    for impl, variants in sorted(opencv_cases.DETECT_MOTION.items()):
        for variant, config in variants:
            outputs = runner.run_detect_motion(impl, config, arrays)
            _record_array_case(group_dir, manifest, "detect_motion", impl,
                               variant, config, input_names, outputs,
                               opencv_cases)


def record_opencv_detectors(group_dir, manifest):
    for impl, spec in sorted(opencv_cases.DETECTORS.items()):
        input_names = spec["inputs"]
        arrays = [imageio_utils.load(input_path(name)) for name in input_names]

        for variant, config in spec["variants"]:
            outputs = runner.run_image_object_detector(impl, config, arrays)

            case_dir = os.path.join(group_dir, "detect", impl, variant)
            os.makedirs(case_dir, exist_ok=True)

            files = {}
            for name, detections in zip(input_names, outputs):
                written = _write_json(
                    os.path.join(case_dir, name + ".json"), detections)
                files[name] = {
                    "file": os.path.relpath(written, group_dir),
                    "detections": len(detections),
                    "sha256": file_digest(written),
                }

            manifest["cases"].append({
                "kind": "detect",
                "impl": impl,
                "variant": variant,
                "config": config,
                "inputs": list(input_names),
                "outputs": files,
                "unstable": {},
            })
            print("  detect {} {} ({} detections)".format(
                impl, variant,
                sum(len(detections) for detections in outputs)))


def _record_arrays_case(group_dir, manifest, kind, impl, variant, config,
                        input_names, results, extra=None):
    """One case whose outputs are named arrays rather than an image."""
    case_dir = os.path.join(group_dir, kind, impl, variant)
    os.makedirs(case_dir, exist_ok=True)

    files = {}
    for name, arrays in zip(input_names, results):
        written = feature_runner.save(os.path.join(case_dir, name), **arrays)
        files[name] = {
            "file": os.path.relpath(written, group_dir),
            "members": {
                member: describe(value)
                for member, value in sorted(arrays.items())
            },
        }

    case = {
        "kind": kind,
        "impl": impl,
        "variant": variant,
        "config": config,
        "inputs": list(input_names),
        "outputs": files,
        "unstable": {},
    }
    case.update(extra or {})
    manifest["cases"].append(case)

    print("  {} {} {} ({} inputs)".format(kind, impl, variant, len(files)))


def record_opencv_features(group_dir, manifest):
    """`detect_features` and `extract_descriptors` of the same name."""
    arrays = [imageio_utils.load(input_path(name))
              for name in feature_cases.IMAGES]

    for impl, variants in sorted(feature_cases.FEATURES.items()):
        for variant, config in variants:
            results = [feature_runner.detect_and_extract(impl, config, array)
                       for array in arrays]
            _record_arrays_case(group_dir, manifest, "features", impl, variant,
                                config, feature_cases.IMAGES, results)


def record_opencv_matches(group_dir, manifest):
    pair = feature_cases.PAIR
    arrays = [imageio_utils.load(input_path(name)) for name in pair]
    name = "_to_".join(pair)

    for impl, variants in sorted(feature_cases.MATCHERS.items()):
        for variant, config in variants:
            for feature_impl in feature_cases.MATCH_FEATURES:
                result = feature_runner.match(
                    impl, config, feature_impl, {}, arrays)

                tag = "{}_{}".format(variant, feature_impl)
                _record_arrays_case(
                    group_dir, manifest, "matches", impl, tag, config,
                    [name], [result], extra={"features": feature_impl})


def record_opencv_tracks(group_dir, manifest):
    pair = feature_cases.PAIR
    arrays = [imageio_utils.load(input_path(name)) for name in pair]
    name = "_to_".join(pair)

    for impl, variants in sorted(feature_cases.TRACKERS.items()):
        for variant, config in variants:
            result = feature_runner.track(impl, config, arrays)
            _record_arrays_case(group_dir, manifest, "tracks", impl, variant,
                                config, [name], [result])


def record_opencv_estimators(group_dir, manifest):
    groups = (
        ("homography", feature_cases.ESTIMATE_HOMOGRAPHY,
         feature_runner.estimate_homography),
        ("fundamental", feature_cases.ESTIMATE_FUNDAMENTAL,
         feature_runner.estimate_fundamental),
    )

    for kind, table, estimate in groups:
        for impl, variants in sorted(table.items()):
            for variant, config in variants:
                for scale in feature_cases.INLIER_SCALES:
                    result = estimate(impl, config, scale)

                    tag = "{}_scale_{:g}".format(variant, scale)
                    _record_arrays_case(
                        group_dir, manifest, kind, impl, tag, config,
                        ["synthetic"], [result],
                        extra={"inlier_scale": scale})


def record_opencv_pipelines(group_dir, manifest):
    for pipeline in opencv_cases.PIPELINES:
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
                for reason in [opencv_cases.unstable_reason(pipeline, "default")]
                if reason
            },
        })
        print("  pipeline {} ({} frames)".format(pipeline, len(files)))


def record_opencv_process_pipelines(group_dir, manifest):
    """Processes recorded through a golden-local pipeline.

    The output is a detected object set rather than an image, so the case
    lands in the same JSON-per-frame shape a detector case has and
    `check_detections` compares it.
    """
    for impl, spec in sorted(opencv_cases.PROCESS_PIPELINES.items()):
        pipeline = spec["pipeline"]

        for variant, settings in spec["variants"]:
            outputs = pipeline_runner.run_detections(pipeline, settings)

            case_dir = os.path.join(group_dir, "process_pipeline", impl,
                                    variant)
            os.makedirs(case_dir, exist_ok=True)

            files = {}
            for name, detections in zip(case_spec.PIPELINE_INPUTS, outputs):
                written = _write_json(
                    os.path.join(case_dir, name + ".json"), detections)
                files[name] = {
                    "file": os.path.relpath(written, group_dir),
                    "detections": len(detections),
                    "sha256": file_digest(written),
                }

            manifest["cases"].append({
                "kind": "process_pipeline",
                "impl": impl,
                "variant": variant,
                "config": {},
                "pipeline": pipeline,
                "settings": list(settings),
                "inputs": list(case_spec.PIPELINE_INPUTS),
                "outputs": files,
                "unstable": {},
            })
            print("  process_pipeline {} {} ({} detections)".format(
                impl, variant,
                sum(len(detections) for detections in outputs)))


def record_opencv(group_dir, manifest):
    record_opencv_filters(group_dir, manifest)
    record_opencv_splits(group_dir, manifest)
    record_opencv_motion(group_dir, manifest)
    record_opencv_detectors(group_dir, manifest)
    record_opencv_features(group_dir, manifest)
    record_opencv_matches(group_dir, manifest)
    record_opencv_tracks(group_dir, manifest)
    record_opencv_estimators(group_dir, manifest)
    record_opencv_refiners(group_dir, manifest)
    record_opencv_warps(group_dir, manifest)
    record_opencv_pipelines(group_dir, manifest)
    record_opencv_process_pipelines(group_dir, manifest)


def record_opencv_warps(group_dir, manifest):
    source = imageio_utils.load(input_path(warp_cases.SOURCE))
    destination = imageio_utils.load(input_path(warp_cases.DESTINATION))
    mask = imageio_utils.load(input_path(warp_cases.MASK))

    for variant, _, _, _ in warp_cases.WARPS:
        warped = warp_runner.run(warp_cases.IMPLEMENTATION, variant,
                                 source, destination, mask)

        _record_array_case(group_dir, manifest, "warp",
                           warp_cases.IMPLEMENTATION, variant, {},
                           [warp_cases.SOURCE], [warped], warp_cases)


def record_opencv_refiners(group_dir, manifest):
    array = imageio_utils.load(input_path(refine_cases.IMAGE))

    for impl, variants in sorted(refine_cases.REFINERS.items()):
        for variant, config in variants:
            result = refine_runner.run(impl, config, array)
            _record_arrays_case(group_dir, manifest, "refine", impl, variant,
                                config, [refine_cases.IMAGE], [result])


def record_measurement_disparity(group_dir, manifest):
    left, right = (imageio_utils.load(input_path(name))
                   for name in measurement_cases.STEREO)

    for impl, variants in sorted(measurement_cases.DISPARITY.items()):
        for variant, config in variants:
            depth = runner.run_stereo_depth_map(impl, config, left, right)
            _record_array_case(group_dir, manifest, "disparity", impl,
                               variant, config, ["stereo"], [depth],
                               measurement_cases)


def record_measurement_targets(group_dir, manifest):
    input_names = measurement_cases.TARGETS
    arrays = [imageio_utils.load(input_path(name)) for name in input_names]

    for impl, variants in sorted(measurement_cases.CALIBRATION_TARGETS.items()):
        for variant, config in variants:
            outputs = runner.run_image_object_detector(impl, config, arrays)

            case_dir = os.path.join(group_dir, "detect", impl, variant)
            os.makedirs(case_dir, exist_ok=True)

            files = {}
            for name, detections in zip(input_names, outputs):
                written = _write_json(
                    os.path.join(case_dir, name + ".json"), detections)
                files[name] = {
                    "file": os.path.relpath(written, group_dir),
                    "detections": len(detections),
                    "sha256": file_digest(written),
                }

            manifest["cases"].append({
                "kind": "detect",
                "impl": impl,
                "variant": variant,
                "config": config,
                "inputs": list(input_names),
                "outputs": files,
                "unstable": {},
            })
            print("  detect {} {} ({} corners)".format(
                impl, variant,
                sum(len(detections) for detections in outputs)))


def record_measurement_calibration(group_dir, manifest):
    pipeline = measurement_cases.CALIBRATION_PIPELINE
    left, right = measurement_cases.calibration_view_names()

    for variant, _ in measurement_cases.CALIBRATION_VARIANTS:
        settings = measurement_cases.calibration_settings(variant)

        outputs = measurement_runner.run_stereo_pipeline(
            pipeline, left, right, settings)

        arrays = measurement_runner.calibration_arrays(outputs)

        _record_arrays_case(group_dir, manifest, "calibration_pipeline",
                            pipeline, variant, {}, ["stereo_rig"], [arrays],
                            extra={"settings": list(settings)})


def record_measurement(group_dir, manifest):
    record_measurement_disparity(group_dir, manifest)
    record_measurement_targets(group_dir, manifest)
    record_measurement_calibration(group_dir, manifest)


def record_calib(group_dir, manifest):
    record_calibrations(group_dir, manifest)
    record_documents(group_dir, manifest)


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
    "calib": record_calib,
    "opencv": record_opencv,
    "measurement": record_measurement,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=sorted(GROUPS),
                        help="recording group")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing recording")
    parser.add_argument("--append", action="store_true",
                        help="record only the cases the manifest does not "
                             "have yet, leaving every existing one alone")
    args = parser.parse_args()

    group_dir = os.path.join(HERE, args.group)
    manifest_path = os.path.join(group_dir, "manifest.json")

    existing = None

    if os.path.exists(manifest_path):
        if args.append:
            with open(manifest_path) as handle:
                existing = json.load(handle)
        elif not args.force:
            print("{} already recorded; pass --force to re-record, or "
                  "--append to add the cases it does not have".format(
                      args.group))
            return 1

    if args.append and existing is None:
        print("{} has no recording to append to".format(args.group))
        return 1

    runner.load_modules()

    if args.group != "calib":
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
        "fixtures": {} if args.group == "calib" else {
            name: describe(imageio_utils.load(input_path(name)))
            for name in sorted(set(case_spec.STILLS) | set(case_spec.SEQUENCE)
                               | set(case_spec.MASKS)
                               | set(case_spec.PIPELINE_INPUTS)
                               | set(opencv_fixtures.build())
                               | set(measurement_fixtures.build()))
        },
        "cases": [],
    }

    GROUPS[args.group](group_dir, manifest)

    if existing is not None:
        # Append only. A recorder must never be run against the code it is
        # meant to be checking, and by the time a group is being extended
        # some of it usually has been replaced -- so an existing case keeps
        # the values it was recorded with, whatever this run produced.
        already = {(case["kind"], case["impl"], case["variant"])
                   for case in existing["cases"]}

        added = [case for case in manifest["cases"]
                 if (case["kind"], case["impl"], case["variant"])
                 not in already]

        for key in manifest:
            if key not in ("cases", "recorded"):
                existing.setdefault(key, manifest[key])

        existing["cases"].extend(added)
        existing["appended"] = datetime.datetime.now(datetime.timezone.utc) \
                                       .strftime("%Y-%m-%dT%H:%M:%SZ")

        print("appended {} case(s); {} already recorded and left alone"
              .format(len(added), len(manifest["cases"]) - len(added)))

        manifest = existing

    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("recorded {} cases into {}".format(len(manifest["cases"]), group_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
