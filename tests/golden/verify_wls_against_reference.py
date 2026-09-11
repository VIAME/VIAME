"""Check the WLS disparity cases against the C++ they stand in for.

Every other case in this framework was recorded by `record.py` while the
implementation it records was still the one in the tree. The three `wls`
variants of `ocv_stereo_disparity` could not be: P7-T06 had already replaced
`plugins/opencv/compute_stereo_disparity.cxx` with
`library/measurement/ocv_stereo_disparity.py` by the time anyone noticed that
the WLS filter -- which the shipped measurement config turns on -- was not
covered, and which is where finding 1.21 lives.

So they were recorded from the python implementation and then checked against
the C++ in the **reference build of `main`**, which still has it. This is that
check, kept so the claim in `measurement_cases.py` can be re-run rather than
taken on trust.

Run it under the reference install, not this one:

    source ~/Dev/viame/build/install/setup_viame.sh
    python tests/golden/verify_wls_against_reference.py

It exits zero when every variant is bit identical.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import imageio_utils                # noqa: E402
import measurement_cases            # noqa: E402
import runner                       # noqa: E402


def main():
    runner.load_modules()

    left, right = (imageio_utils.load(
                       os.path.join(HERE, "inputs", name + ".png"))
                   for name in measurement_cases.STEREO)

    worst = 0.0

    for variant, config in measurement_cases.DISPARITY["ocv_stereo_disparity"]:
        if not variant.startswith("wls"):
            continue

        got = np.squeeze(runner.run_stereo_depth_map(
            "ocv_stereo_disparity", config, left, right))

        recorded = imageio_utils.load(os.path.join(
            HERE, "measurement", "disparity", "ocv_stereo_disparity", variant,
            "stereo.npz"))

        assert got.shape == recorded.shape, (
            "{}: {} against a recorded {}".format(
                variant, got.shape, recorded.shape))
        assert got.dtype == recorded.dtype, (
            "{}: {} against a recorded {}".format(
                variant, got.dtype, recorded.dtype))

        difference = np.abs(got.astype(np.float64) -
                            recorded.astype(np.float64))
        worst = max(worst, float(difference.max()))

        print("{:14s} {} {} max difference {}".format(
            variant, got.dtype, got.shape, difference.max()))

    print("worst difference over every variant: {}".format(worst))

    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
