"""What the calibration golden recording covers.

Two contracts, recorded separately because phase 7 replaces them in different
tasks.

`calibration` is what `viame::read_stereo_rig` -- reached through
`viame.core._measurement.load_stereo_calibration`, which is what `convert_cam`
and the measurement pipelines use -- makes of each calibration source. That is
the contract every VIAME caller sees, and it has to survive P7-T05 and
P7-T06 unchanged.

`nodes` is one level below: every node of each OpenCV YAML and XML file, as
`cv::FileStorage` parses it. That is what `library/file_io/opencv_yaml` in
P7-T05 has to reproduce, including the nodes VIAME does not currently read --
`R1`, `R2`, `P1`, `P2` and `Q` in `extrinsics.yml` are written by the
calibration processes and read by nothing in C++ yet, so a reader that
silently dropped them would pass every other check.
"""

# Calibration sources `load_stereo_calibration` accepts, by fixture name.
# Paths are relative to the repository root.
#
# Between them these cover four of the five formats `read_stereo_rig`
# dispatches on: an OpenCV directory pair, a single OpenCV YAML, and two NPZ.
# JSON and MAT have no committed fixture; P7-T05 adds one if it grows a
# reader change that would touch them.
CALIBRATIONS = {
    "tests_data_dir":       "tests/data",
    "tests_data_yml":       "tests/data/intrinsics.yml",
    "ifremer_dir":          "configs/add-ons/ifremer/models",
    "ifremer_npz":          "configs/add-ons/ifremer/models/calibration.npz",
    "size_measurement_npz": "examples/size_measurement/calibration_matrices.npz",
}

# The arrays `load_stereo_calibration` returns.
CALIBRATION_KEYS = ("k_left", "dist_left", "k_right", "dist_right",
                    "rotation", "translation")


# FileStorage documents, by fixture name. The two YAML shapes VIAME writes
# and reads, plus the XML shape: `Model_SVM.xml` is the hierarchical SVM's
# index, which `classify_fish_hierarchical_svm` reads with FileStorage and
# which is the only XML VIAME parses itself -- the per class `speciesSVM_*`
# files go straight to `cv::ml::SVM::load`.
DOCUMENTS = {
    "intrinsics_yml":   "tests/data/intrinsics.yml",
    "extrinsics_yml":   "tests/data/extrinsics.yml",
    "ifremer_intrinsics_yml": "configs/add-ons/ifremer/models/intrinsics.yml",
    "ifremer_extrinsics_yml": "configs/add-ons/ifremer/models/extrinsics.yml",
    "model_svm_xml":    "configs/add-ons/uw-fish/models/Model_SVM.xml",
}

# Values are compared exactly: these are text files holding decimal
# literals, and a parser that rounds differently is a parser that is wrong.
TOLERANCE = 0.0


# The `nodes` recording is taken through python's `cv2.FileStorage`, which is
# OpenCV itself. That is the right reference -- it is the definition of the
# format -- and it stays available after phase 7, because python keeps `cv2`
# (open decision 1). It is not, on its own, a test of the replacement:
# P7-T05 writes `library/file_io/opencv_yaml.{h,cxx}` in C++, and this case
# has to be replayed through that too once it exists and is reachable, the
# way `calibration` already goes through VIAME's own reader. Until then the
# recording is the specification and nothing is checking the C++ side.
