# The Eigen recording

`eigen.json` is what Eigen computed for a fixed set of inputs, recorded by
`record_from_eigen.cxx` while Eigen was still a dependency of the build.
`library/core_types`' matrix, rotation and transform types replaced it in
phase 6, and this is what they are held to.

The test that reads it is `tests/library/core_types/test_math.cxx`, beside
the code it checks; the directory it reads from is passed in as
`VIAME_GOLDEN_MATH_DIR`. Only the recording and the recorder are here, for
the same reason the OpenCV recorders are under `../opencv/`: a file that
reaches for a reference implementation belongs with that reference.

Re-recording needs an Eigen that the build no longer has, which is the point
-- the recording outlives the dependency. `tests/reference/README.md` has the
general shape.
