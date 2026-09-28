# Phase 12: what the assessment found

Not in the original plan. These are the tasks `lite-completion.md` section 5.3
ranks, written in the form the other phases use so that they can be picked
up the same way. They are independent of each other unless `Depends` says
otherwise, and none of them removes a dependency: the dependencies are gone.
What is left is making sure that is as true as it looks.

References: lite-completion.md §4, §5.

### P12-T01 Configure every shipped pipeline
Depends: -
Do:
- Extend `pipe-check` with `--configure`: after baking, construct each process and call its `_configure`, without running a step. A process whose configuration needs a file that is not in the install (a model, a calibration) reports `needs <path>` rather than failing.
- Record the result per pipeline in `tests/baseline/pipes.json` beside `status`, as `configures`: `ok`, `needs`, or the exception text.
- `baseline:pipes` compares it. A pipeline that configured and stops configuring is a failure; one that starts is reported, as a status change is now.
Done when:
- Reverting `f5072a5c6` turns `baseline:pipes` red on the five pipelines that set `refine_disparity_segment`.
- The count of pipelines that configure is recorded in STATUS.md. It will not be 258.

### P12-T02 Install the python dependencies from the regenerated locks
Depends: -
Do:
- In a fresh prefix, configure with `VIAME_INSTALL_PYTHON_DEPS=ON` and build. Do it twice, for `py3.10/cuda12.lock` and `py3.12/cuda13.lock`.
- Record which OpenCV distribution is left in `cv2/` and whether `import cv2`, `import ultralytics`, `import albumentations`, `import mmdet` and `import kwimage` succeed.
- Run `ctest -L "BASELINE|UNIT|CORE|GOLDEN|CRITICAL"` against that install.
Done when:
- Both installs pass, or every failure is in STATUS.md with its cause.
- `pip check` output is recorded; it is expected to complain about the two OpenCV distributions and nothing else.

### P12-T03 Move `kwimage` to a version that carries its own fix
Depends: P12-T02
Do:
- `kwimage[headless]>=0.12.0` in `base.in` and `cmake/wheel/requirements.txt`; regenerate all sixteen locks.
- Delete `cmake/packaging/patches/apply.py`, which nothing calls.
- A test that draws text on a float32 image through `kwimage.draw_text_on_image`.
Done when:
- The test passes on a fresh install and fails with `kwimage==0.11.6`.
- `git grep packaging/patches/apply` is empty.

### P12-T04 Check every fork
Depends: -
Do:
- `baseline:fork_cv2` fails, rather than printing a list, when an enabled fork's submodule is not checked out. A fork that is not enabled is skipped and named.
- Check out the thirteen that are not: `sam3`, `torchvision`, `torchvideo`, `sleap-nn`, `mit-yolo`, `litdet`, `foundation-stereo`, `fast-foundation-stereo`, `dino3`, `detectron2`, `darknet-to-pytorch-onnx`, `pyav`. Record what each imports.
- For each that imports `cv2` on a path VIAME reaches, a patch in `packages/patches/<fork>.patch` against VIAME's own implementations. `sam3` is known to need one: eight files, `findContours`, `drawContours`, `distanceTransform`, `dilate`, `imread`, `resize`, `putText`, `VideoWriter`.
Done when:
- `baseline:fork_cv2` passes with every enabled fork checked out.
- STATUS.md lists, per fork, the number of files that imported `cv2` and the number patched.

### P12-T05 Finish the names
Depends: -
Do:
- `library/file_io/opencv_yaml.{h,cxx}` and its binding become `file_storage`; `viame.file_io._opencv_yaml` becomes `viame.file_io._file_storage`. The strings `opencv_storage` and `!!opencv-matrix` do not change: they are the file format.
- Delete `library/utilities/compat/opencv/`. No pipeline, example or tool imports through it.
- `tests/reference/opencv_{cases,fixtures}.py` follow the group they serve.
- Prose in `library/`, `tools/`, `cmake/`, `examples/` and `docs/`: say what the code does. A statement that a kernel reproduces a reference exactly, over how many inputs, stays as a statement about the reference tests in `tests/reference/opencv/`; it is not deleted.
- Not touched: the `ocv_*` implementation names and the files named for them, `design/`, `tests/reference/opencv/`, `library/tpl/`, `RELEASE_NOTES.md`.
Done when:
- `git grep -il "opencv\|cv2" -- library tools cmake examples docs` lists only files whose name begins `ocv_` and the two baseline guards.
- `ctest -L "BASELINE|UNIT|CORE|GOLDEN|CRITICAL"` passes.

### P12-T06 Torch and torchvision from source
Depends: P12-T02
Do:
- Restore `VIAME_BUILD_PYTORCH_FROM_SOURCE` and `VIAME_BUILD_TORCHVISION_FROM_SOURCE` as options. `main`'s recipe is `cmake/add_project_pytorch.cmake`, 745 lines; `TORCH_CUDA_ARCH_LIST` takes this tree's `CUDA_ARCHITECTURES` unchanged.
- OpenBLAS: `main` took it from fletch. Decide between a system package and building it, and record the choice.
- When either flag is on, torch and torchvision must not be installed from the lock. `viame_python_deps.cmake` installs a lock whole, so this is either a filter on the installed lock or accelerator `.in` files without torch in them.
Done when:
- A build with both flags on, targeting a CUDA the index does not publish, imports a torch whose `torch.version.cuda` is that CUDA.
- `pip list` shows one torch.

### P12-T07 Installs that do not accumulate
Depends: -
Do:
- The install manifest records what this build installs. At install time, remove from the prefix any file a previous VIAME manifest listed and this one does not. Files VIAME never installed are not touched.
- `baseline:install` fails on a file in the prefix that is under a VIAME-owned directory and in no manifest.
Done when:
- An install over the reference prefix removes the 18 libraries from before the fold and nothing else.

### P12-T08 A colour recording wider than a vector
Depends: -
Do:
- `tests/reference/opencv/recorders/record_image_kernels.py` records the colour conversions over a window 100 wide as well as 32.
- The lane count the kernel assumes becomes a named constant with the build it corresponds to beside it.
Done when:
- `test_color.cxx` exercises `i >= ( width / 32 ) * 32` and passes.
- Setting the constant to 16 fails it.
