# OpenCV

Everything whose subject is OpenCV: the recordings taken from the `ocv_*`
implementations, the live comparisons against `cv2`, and the recorders that
wrote the recordings in the sibling groups. `../README.md` says how a
reference test works in general; this file is the detail.

This is the only place in the tree that imports `cv2`.

## The recordings

`manifest.json` and the directories beside it are the `opencv` group -- 124
cases over the filters, splits, warps, the SIFT/SURF/FLANN feature chain, the
detectors and refiners, and the pipelines that use them. They are replayed by
`../test_golden.py` like every other group and need nothing installed.

## The live comparisons

`test_kernels_*.py` call `cv2` and VIAME on the same input and compare. They
are the kernels whose whole specification is "the same numbers as OpenCV" --
the parallel fast paths and the streaming arrangements, where VIAME computes
a row band at a time and OpenCV computes the image whole, so agreement is
not obvious and is worth a test.

Most of `library/image_kernels` is **not** checked this way. It is checked
against `../image_kernels/opencv.json`, a recording, because a recording
outlives the thing it records. These two are live because a rolling sum and
a thread pool have too many configurations to commit.

## Skipping

Every module begins with `pytest.importorskip('cv2')`. A tree without it
skips rather than fails, which is the ordinary case now -- VIAME declares no
imaging library, so `cv2` arrives only because `ultralytics`,
`albumentations` and `mmengine` ask for it. A module that needs a `contrib`
build says so separately and skips on `hasattr`.

## `recorders/`

Developer tools, not tests. They run once, by hand, against `cv2`, and write
into `../`, where the golden replay reads them afterwards without
needing `cv2` at all:

| Script | Writes |
|---|---|
| `record_image_kernels.py` | `../image_kernels/opencv.json` |
| `record_projection.py` | `../projection/opencv.json` |
| `record_sift.py` | `../sift/opencv.json` |
| `record_surf.py` | `../surf/opencv.json` and its tiles |
| `file_storage_reference.py` | imported by `../record.py` for the `nodes` case |

`record_surf.py` is the one that cannot be re-run here: SURF is patented, no
wheel on PyPI is built with `OPENCV_ENABLE_NONFREE`, and VIAME implements it
itself for that reason (`library/image_processing/surf.h`). Its recording
came from a build that has it.
