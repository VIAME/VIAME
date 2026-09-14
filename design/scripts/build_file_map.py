#!/usr/bin/env python3
"""Propose a destination under `library/` for every file under `plugins/`.

Phase 2 moves `plugins/` -- which is organised by *dependency*, so that
everything needing torch sits in `plugins/pytorch` and everything needing
OpenCV in `plugins/opencv` -- into `library/<capability>/`, organised by what
the code does. The two cuts cross: `plugins/core` alone holds readers,
processes, measurement, descriptors, training and evaluation, and
`plugins/pytorch` holds detectors, trackers, segmenters and trainers.

So the mapping cannot come from the path. It comes, in order of how much it
is worth trusting:

1. **What a C++ algorithm registers.** `register_algorithms.cxx` names the
   interface and the class for every one, and the class name is the file
   name. An interface belongs to exactly one capability.
2. **What a python package declares.** P8-T10 turned every implementation
   into `( interface, name, "module:Class" )`, so the same interface table
   answers for python.
3. **What a C++ process registers.** `register_processes.cxx` names the
   process and instantiates its class.
4. **Vendored subtrees**, which move whole: netharn, siammask, srnn and the
   rest are somebody else's model code sitting under the plugin that wrapped
   it.
5. **Names**, last and least, for the leftovers.

Anything none of those reach is reported as UNMAPPED rather than guessed at.

Usage:
    build_file_map.py [--out design/lite-file-map.tsv]
"""

import argparse
import ast
import os
import re
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


# An algorithm interface belongs to one capability. From
# `lite-library-layout.md` section 1.
INTERFACE_DIR = {
    "image_object_detector": "object_detectors",
    "detect_motion": "object_detectors",
    "train_detector": "training",
    "train_tracker": "training",
    "track_objects": "object_trackers",
    "initialize_object_tracks": "object_trackers",
    "associate_detections_to_tracks": "object_trackers",
    "compute_association_matrix": "object_trackers",
    "refine_detections": "classifiers",
    "refine_tracks": "classifiers",
    "merge_detections": "classifiers",
    "detected_object_filter": "classifiers",
    "segment_via_points": "segmentation",
    "compute_stereo_depth_map": "measurement",
    "detected_object_set_input": "file_io",
    "detected_object_set_output": "file_io",
    "read_object_track_set": "file_io",
    "write_object_track_set": "file_io",
    "read_track_descriptor_set": "file_io",
    "write_track_descriptor_set": "file_io",
    "transform_2d_io": "file_io",
    "image_io": "video_io",
    "video_input": "video_io",
    "image_filter": "image_processing",
    "split_image": "image_processing",
    "merge_images": "image_processing",
    "warp_image": "image_processing",
    "compute_track_descriptors": "descriptors",
    "perform_query": "descriptors",
    "perform_text_query": "descriptors",
    "query_track_descriptor_set": "descriptors",
    "detect_features": "image_processing",
    "extract_descriptors": "image_processing",
    "match_features": "image_processing",
    "estimate_homography": "image_processing",
    "estimate_fundamental_matrix": "image_processing",
    "optimize_cameras": "measurement",
    "triangulate_landmarks": "measurement",
    "close_loops": "image_processing",
    "track_features": "image_processing",
}


# Subtrees that move whole, because they are vendored model code rather than
# VIAME's own implementations.
SUBTREE_DIR = {
    "plugins/pytorch/netharn": "training",
    "plugins/pytorch/learn": "training",
    "plugins/pytorch/onnx_exporters": "training",
    "plugins/pytorch/remax": "object_detectors",
    "plugins/pytorch/torchvision": "object_detectors",
    "plugins/pytorch/detectron2": "object_detectors",
    "plugins/pytorch/siammask": "object_trackers",
    "plugins/pytorch/srnn": "object_trackers",
    "plugins/pytorch/mdnet": "object_trackers",
    "plugins/pytorch/minima_loftr": "image_processing",
    "plugins/templates": "examples",
    "plugins/examples": "examples",
    "plugins/claude": "utilities",
    "plugins/seagis": "measurement",
    "plugins/colmap": "measurement",
    "plugins/vertex-ai": "object_detectors",
    "plugins/svm": "descriptors",
}


# The leftovers, by name. Every entry here is a judgement rather than
# evidence, which is why the list is written out rather than inferred.
# Per-directory structural files. These are not moved: phase 2 writes one
# `CMakeLists.txt`, one `register.cxx` and one `__init__.py` per library, and
# the contents of the old ones are distributed by the moves rather than
# carried. `apply_file_map.py` leaves them alone and the phase deletes them
# when the directory is empty.
STRUCTURAL = re.compile(
    r"(^|/)(CMakeLists\.txt|register_(algorithms|processes)\.(cxx|h)"
    r"|__init__\.py|README(\.md|\.txt|\.rst)?|\.gitignore)$")


NAME_DIR = [
    (r"^evaluate_models", "evaluation"),
    (r"^(applet_attributes|atomic_output|utilities_file|model_wrap|compat|types)$",
     "utilities"),
    (r"^(survey_metadata|utilities_ply)$", "utilities"),
    (r"^(detection_fusion_core|utilities_target_clfr)$", "classifiers"),
    (r"^(interactive_service|utility_processes)$", "utilities"),
    (r"^convert_notes_to_attributes$", "file_io"),
    (r"^(geometry_numpy|prior_coverage_opencv)$", "measurement"),
    (r"^onnx_(clf_)?predictor$", "object_detectors"),
    (r"^(convert_color_space|debayer_filter|random_hue_shift|fft_filter_based_on_ref)$",
     "image_processing"),
    (r"^windowed_utils$", "object_detectors"),
    (r"^(mmdet_launcher|rf_detr_launcher|sleap_launcher|sleap_common"
     r"|mmdet_compatibility|convert_to_onnx_process)$", "training"),
    (r"^(dino_matcher|torchvision_augment_process)$", "object_detectors"),
    (r"^sam3_utilities$", "segmentation"),
    (r"^(manipulate_pipelines|python_script_applet|compat|types|utilities|utils)$",
     "utilities"),
    (r"^(camera_io|camera_rig_io|read_transform|write_transform|convert_annotations)",
     "file_io"),
    (r"^(filename_to_timestamp|add_timestamp_from_filename|detect_shot_breaks)",
     "video_io"),
    (r"stereo|calibrat|measure|disparity|triangulat|epipolar|rectif", "measurement"),
    (r"descriptor|iqr|query", "descriptors"),
    (r"train|adaptive_", "training"),
    (r"track", "object_trackers"),
    (r"detector|detect_", "object_detectors"),
    (r"refine|merge_detections|classif", "classifiers"),
    (r"mask|polygon|segment|keypoint", "segmentation"),
    (r"image|frame|homog|mosaic|align|warp|registration|optical_flow", "image_processing"),
    (r"csv|json|coco|dive|cvat|habcam|oceaneyes|fishnet|yolo|kw18|read_|write_", "file_io"),
]


def tracked_files():
    out = subprocess.run(["git", "ls-files", "plugins"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return [f for f in out if "__pycache__" not in f]


def cxx_algorithm_interfaces():
    """{class name: interface} from every register_algorithms.cxx."""
    found = {}
    pattern = re.compile(
        r"register_algorithm<\s*(?:kv::algo::|kwiver::vital::algo::)?([a-z_0-9]+)\s*,"
        r"\s*([a-z_0-9:]+)\s*>", re.S)

    for path in subprocess.run(["git", "ls-files", "plugins/*/register_algorithms.cxx"],
                               cwd=ROOT, capture_output=True, text=True).stdout.split():
        for interface, klass in pattern.findall(open(os.path.join(ROOT, path)).read()):
            found[klass.split("::")[-1]] = interface
    return found


def cxx_process_classes():
    """Every class registered as a process, by name."""
    found = set()
    pattern = re.compile(r"create_new_process<\s*([a-z_0-9:]+)\s*>")
    for path in subprocess.run(["git", "ls-files", "plugins/*/register_processes.cxx"],
                               cwd=ROOT, capture_output=True, text=True).stdout.split():
        for klass in pattern.findall(open(os.path.join(ROOT, path)).read()):
            found.add(klass.split("::")[-1])
    return found


def python_declarations():
    """{module path: interface} from the P8-T10 declarations."""
    found = {}
    for path in subprocess.run(["git", "ls-files", "plugins/*/__init__.py"],
                               cwd=ROOT, capture_output=True, text=True).stdout.split():
        tree = ast.parse(open(os.path.join(ROOT, path)).read())
        for node in tree.body:
            if not (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)):
                continue
            kind = node.targets[0].id
            if kind == "__vital_algorithm_declarations__":
                for element in node.value.elts:
                    interface, _name, _desc, import_path = ast.literal_eval(element)
                    found[import_path.split(":")[0]] = interface
            elif kind == "__sprokit_process_declarations__":
                for element in node.value.elts:
                    _name, _desc, import_path = ast.literal_eval(element)
                    found.setdefault(import_path.split(":")[0], None)
    return found


def module_to_path(module):
    """`viame.core.optical_flow` -> `plugins/core/optical_flow.py`."""
    parts = module.split(".")
    assert parts[0] == "viame", module
    return "plugins/" + "/".join(parts[1:]) + ".py"


def classify(path, algo_interfaces, process_classes, declared):
    stem = os.path.splitext(os.path.basename(path))[0]

    # A vendored package -- `plugins/pytorch/netharn` and friends, three
    # components deep -- moves whole and keeps its internal structure: those
    # paths are its import paths, and `remax/model/utils.py` and
    # `remax/util/utils.py` are two different modules. Its own `__init__.py`
    # and `CMakeLists.txt` are part of the package, not VIAME's registration
    # files, so this is checked before the structural rule rather than after.
    for subtree, directory in SUBTREE_DIR.items():
        if subtree.count("/") == 2 and path.startswith(subtree + "/"):
            leaf = os.path.basename(subtree)
            relative = path[len(subtree) + 1:]
            return directory, "subtree:%s/%s" % (leaf, relative)

    if STRUCTURAL.search(path):
        return "STRUCTURAL", "structural"

    # A whole plugin that becomes one library directory flattens into it.
    for subtree, directory in SUBTREE_DIR.items():
        if subtree.count("/") == 1 and path.startswith(subtree + "/"):
            return directory, "subtree" 

    if stem in algo_interfaces:
        interface = algo_interfaces[stem]
        if interface in INTERFACE_DIR:
            return INTERFACE_DIR[interface], "registers " + interface

    if path in declared:
        interface = declared[path]
        if interface and interface in INTERFACE_DIR:
            return INTERFACE_DIR[interface], "declares " + interface

    process_stem = stem[:-len("_process")] if stem.endswith("_process") else stem
    if stem in process_classes or process_stem in algo_interfaces:
        pass  # falls through to the name rules, which know about processes

    for pattern, directory in NAME_DIR:
        if re.search(pattern, stem):
            return directory, "name"

    return None, None


# When two plugins hold a file of the same name, they hold two different
# implementations -- `plugins/core/pair_stereo_detections` registers
# `pair_stereo_detections` and `plugins/opencv/pair_stereo_detections`
# registers `ocv_pair_stereo_detections`. The dependency-named directory was
# doing the disambiguating, and once it is gone the file name has to. The
# prefix is the one the implementation already registers under.
COLLISION_PREFIX = {
    "opencv": "ocv_",
    "onnx": "onnx_",
    "pytorch": "torch_",
    "svm": "svm_",
    "seagis": "seagis_",
    "colmap": "colmap_",
    "vertex-ai": "vertex_ai_",
}


def disambiguate(rows):
    """Prefix a destination whose basename more than one source claims."""
    claimed = {}
    for source, destination in rows:
        claimed.setdefault(destination, []).append(source)

    resolved, renamed = [], 0
    for source, destination in rows:
        if destination == "STRUCTURAL" or len(claimed[destination]) == 1:
            resolved.append((source, destination))
            continue

        plugin = source.split("/")[1]
        prefix = COLLISION_PREFIX.get(plugin, "")
        if not prefix:
            resolved.append((source, destination))
            continue

        directory, base = destination.rsplit("/", 1)
        if base.startswith(prefix):
            resolved.append((source, destination))
        else:
            resolved.append((source, "%s/%s%s" % (directory, prefix, base)))
            renamed += 1

    return resolved, renamed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="design/lite-file-map.tsv")
    args = parser.parse_args()

    algo_interfaces = cxx_algorithm_interfaces()
    process_classes = cxx_process_classes()
    declared = {module_to_path(m): i for m, i in python_declarations().items()}

    print("evidence: %d C++ algorithms, %d C++ processes, %d declared python modules"
          % (len(algo_interfaces), len(process_classes), len(declared)))

    rows, unmapped = [], []
    reasons = {}

    for path in tracked_files():
        directory, why = classify(path, algo_interfaces, process_classes, declared)
        if directory is None:
            unmapped.append(path)
            continue
        if directory == "STRUCTURAL":
            destination = "STRUCTURAL"
        elif why.startswith("subtree:"):
            destination = "library/%s/%s" % (directory, why.split(":", 1)[1])
        else:
            destination = "library/%s/%s" % (directory, os.path.basename(path))
        rows.append((path, destination))
        key = why.split(":")[0].split()[0]
        reasons[key] = reasons.get(key, 0) + 1

    rows, renamed = disambiguate(rows)
    if renamed:
        print("   %-12s %d destinations prefixed to break a name collision"
              % ("collision", renamed))

    out = os.path.join(ROOT, args.out)
    with open(out, "w") as handle:
        handle.write("# source\tdestination\n")
        for source, destination in sorted(rows):
            handle.write("%s\t%s\n" % (source, destination))

    total = len(rows) + len(unmapped)
    print("mapped %d of %d (%.0f%%) into %s"
          % (len(rows), total, 100.0 * len(rows) / total, args.out))
    for why, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("   %-12s %d" % (why, count))

    if unmapped:
        print("\nUNMAPPED (%d):" % len(unmapped))
        for path in unmapped[:40]:
            print("   " + path)
        if len(unmapped) > 40:
            print("   ... and %d more" % (len(unmapped) - 40))

    return 0


if __name__ == "__main__":
    sys.exit(main())
