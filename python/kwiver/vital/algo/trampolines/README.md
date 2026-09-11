# Hand-written algorithm trampolines

`cpp_to_pybind11.py` generates one trampoline per algorithm interface, and
every generated method is a `PYBIND11_OVERLOAD` that passes its arguments to
python by value. That is right for an input and wrong for an in/out
parameter: a python implementation can fill one all it likes and the C++
caller sees nothing.

Three interfaces in this tree have one, and all three are on the path phase 7
moves to python:

| Interface | The parameter | What it costs |
|---|---|---|
| `extract_descriptors` | `feature_set_sptr& features` | The extractor may reorder or drop features to line up with its descriptors; a caller left holding the old set pairs each descriptor with the wrong feature |
| `estimate_homography` | `std::vector<bool>& inliers` | `match_features_homography` and `compute_ref_homography_core` keep only the inlier matches, so an empty vector keeps none |
| `estimate_fundamental_matrix` | `std::vector<bool>& inliers` | `match_features_fundamental_matrix`, the same way |

A file here named `<interface>_trampoline.txx` is copied in place of the
generated one. The convention it establishes, for these three and for any
later one: **the python method returns a tuple of the return value followed
by the out parameters, in the order the C++ signature declares them**, and
the trampoline writes them back. A python implementation that returns the
bare value still works -- the trampoline leaves the out parameter alone and
logs -- because that is what an implementation written before this existed
does, and a silent tuple requirement would be its own trap.

Callers are unaffected: the *calling* side of these methods is bound
separately (see `extract_descriptors_extras.cxx` and
`estimator_extras.cxx`) and keeps its own signature. Only implementing one
in python is what this changes.

## One python name, two C++ overloads

Both estimators declare `estimate` twice: once taking point lists, which is
pure virtual, and once taking feature sets and a match set, which has a C++
body that reduces to the first. Python has one name for the two, so the
generated trampoline sent both to whatever `estimate` the implementation
defined -- and an implementation that reasonably wrote the pure one got
called with the other one's arguments.

So the second overload looks for `estimate_matches`, the name
`estimator_extras.cxx` gives it on the calling side, and falls back to the
C++ body when python does not define it. A python implementation therefore
writes `estimate` alone and gets the feature-set form for free, which is what
the C++ ones do.

P8-T02 replaces the generator with hand-written trampolines throughout. This
is that task's first three files, brought forward because P7-T04 cannot move
SIFT, SURF and the estimators to python without them.
