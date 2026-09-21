#!/usr/bin/env python3
"""Tests for `build_wheel.py`.

    python3 cmake/wheel/test_build_wheel.py      # no pytest needed
    pytest cmake/wheel/test_build_wheel.py

The glob translation carries most of the risk here, and one case in it is not
hypothetical: the first version derived a destination from the glob's leading
literal directories, which stop at the first wildcard. With
`lib/python3.*/site-packages/kwiver/**` that prefix is `lib`, so every
destination kept `python3.10/site-packages/` on the front and the wheel
installed a `kwiver` package whose contents were another `kwiver` package. It
built, it passed a size check, and it failed at `import kwiver.vital`. The
`tail_*` cases below are that bug written down.
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_wheel as bw  # noqa: E402


# ----------------------------------------------------------------------------
# Glob matching
# ----------------------------------------------------------------------------

def test_star_does_not_cross_separators():
    assert bw._match("lib/*.so", "lib/a.so")
    assert not bw._match("lib/*.so", "lib/python3.10/a.so")


def test_doublestar_crosses_separators():
    assert bw._match("lib/**", "lib/a/b/c.so")
    assert bw._match("**/__pycache__/**", "a/b/__pycache__/c.pyc")


def test_anchored_at_both_ends():
    # `viame` must not also match `viame_extra`
    assert not bw._match("lib/python3.*/site-packages/viame/**",
                         "lib/python3.10/site-packages/viame_extra/a.py")
    assert not bw._match("lib/libviame*.so*", "lib/python3.10/libviame.so")


def test_tail_is_what_doublestar_consumed():
    ok, tail = bw._match_tail("lib/python3.*/site-packages/viame/**",
                              "lib/python3.10/site-packages/viame/a/b.py")
    assert ok and tail == "a/b.py"


def test_tail_with_wildcard_before_the_doublestar():
    """The regression: a wildcard mid-path must not leak into the tail."""
    ok, tail = bw._match_tail("lib/python3.*/site-packages/kwiver/**",
                              "lib/python3.10/site-packages/kwiver/vital/types/x.so")
    assert ok
    assert tail == "vital/types/x.so", f"tail leaked the wildcard segment: {tail!r}"


def test_tail_is_none_without_a_doublestar():
    ok, tail = bw._match_tail("lib/libviame*.so*", "lib/libviame.so.1.0.0")
    assert ok and tail is None


# ----------------------------------------------------------------------------
# Selection and packing, against a prefix built for the purpose
# ----------------------------------------------------------------------------

def _prefix(tmp):
    root = Path(tmp)
    for rel in ("lib/python3.10/site-packages/pkg/__init__.py",
                "lib/python3.10/site-packages/pkg/sub/mod.so",
                "lib/python3.10/site-packages/pkg/__pycache__/x.pyc",
                "lib/python3.10/site-packages/pkg/tests/test_a.py",
                "lib/libthing.so.1",
                "lib/libthing.a",
                "include/thing.h"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    return root


CONTENTS = """
include lib/python3.*/site-packages/pkg/** -> pkg/
include lib/libthing*.so*                  -> {data}/lib/
exclude **/__pycache__/**
exclude **/tests/**
exclude **/*.a
"""


def _select(tmp):
    root = _prefix(tmp)
    cf = root / "contents.txt"
    cf.write_text(CONTENTS)
    return bw.select(root, bw.read_contents(cf), "d-1.0.data/data")


def test_selection_maps_and_excludes(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        chosen = _select(tmp)
        assert "pkg/__init__.py" in chosen
        assert "pkg/sub/mod.so" in chosen
        assert "d-1.0.data/data/lib/libthing.so.1" in chosen
        # excluded
        assert not any("__pycache__" in d for d in chosen)
        assert not any(d.endswith("test_a.py") for d in chosen)
        assert not any(d.endswith(".a") for d in chosen)
        # not selected at all
        assert not any("thing.h" in d for d in chosen)


def test_wheel_is_a_valid_zip_with_a_record(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = _prefix(tmp)
        cf = root / "contents.txt"
        cf.write_text(CONTENTS)
        out = Path(tmp) / "out"
        bw.main(["--prefix", str(root), "--contents", str(cf),
                 "--output-dir", str(out), "--name", "d", "--version", "1.0",
                 "--top-level", "pkg"])
        whl = next(out.glob("*.whl"))
        with zipfile.ZipFile(whl) as z:
            assert z.testzip() is None
            names = z.namelist()
            assert "d-1.0.dist-info/RECORD" in names
            assert "d-1.0.dist-info/METADATA" in names
            assert "d-1.0.dist-info/WHEEL" in names
            record = z.read("d-1.0.dist-info/RECORD").decode()
            # every packed file is listed, and RECORD lists itself bare
            for n in names:
                assert n in record, f"{n} missing from RECORD"
            assert "d-1.0.dist-info/RECORD,," in record
            assert "Root-Is-Purelib: false" in z.read("d-1.0.dist-info/WHEEL").decode()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"  {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
