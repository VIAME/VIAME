"""What `library/file_io/opencv_yaml` guarantees beyond the golden replay.

`tests/golden/test_golden.py`'s `nodes` case holds the reader to what
`cv::FileStorage` parsed out of every calibration document VIAME ships. What
is here is the rest of the contract:

* the writer produces the **same bytes** FileStorage did, which is what makes
  a regenerated calibration file diff cleanly against the one beside it;
* read, write, read again gives the same document;
* the corners of the format that no shipped fixture happens to have.

Run just these:  ctest -R "unit:file_io"
"""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GOLDEN = os.path.abspath(os.path.join(HERE, "..", "..", "golden"))

sys.path.insert(0, GOLDEN)


@pytest.fixture(scope="module")
def yaml():
    from viame.file_io import _opencv_yaml
    return _opencv_yaml


# Every OpenCV document committed in the tree, by the name the golden gives
# it. The YAML ones are what the writer is held to; the XML one is read only.
def documents():
    import calib_cases
    return calib_cases.DOCUMENTS


YAML_DOCUMENTS = sorted(name for name, source in documents().items()
                        if source.endswith((".yml", ".yaml")))


@pytest.mark.parametrize("name", YAML_DOCUMENTS)
def test_writing_reproduces_opencv_byte_for_byte(yaml, name):
    """The committed file is the recording here.

    Every one of these was written by `cv::FileStorage`, so reading one and
    writing it back is a comparison against OpenCV's own output -- including
    where a long data array wraps, which is the part a reimplementation is
    most likely to get nearly right.
    """
    source = os.path.join(REPO, documents()[name])

    with open(source) as handle:
        original = handle.read()

    assert yaml.to_yaml(yaml.read(source)) == original


@pytest.mark.parametrize("name", sorted(documents()))
def test_reading_a_written_document_gives_the_same_document(yaml, name, tmp_path):
    source = os.path.join(REPO, documents()[name])
    parsed = yaml.read(source)

    # XML is read only, so its round trip goes out through YAML. The values
    # have to survive the change of syntax, which for `Model_SVM.xml` means
    # a sequence of maps written in the block form.
    written = tmp_path / "round_trip.yml"
    yaml.write(str(written), parsed)

    assert yaml.read(str(written)) == parsed


# ----------------------------------------------------------------------------
# How a number is spelled
# ----------------------------------------------------------------------------
#
# `icvDoubleToString`: a value equal to its own rounding is written as the
# integer and a full stop, and everything else as %.16e. That rule is why a
# calibration file is full of bare `0.` and why a reimplementation that used
# %g everywhere would produce a file that still parses and never matches.

@pytest.mark.parametrize("value,spelling", [
    (0.0, "0."),
    (1.0, "1."),
    (-3.0, "-3."),
    (639.5, "6.3950000000000000e+02"),
    (0.42357378893910225, "4.2357378893910225e-01"),
])
def test_doubles_are_spelled_as_opencv_spells_them(yaml, value, spelling,
                                                   tmp_path):
    path = tmp_path / "numbers.yml"
    yaml.write(str(path), {"M": {"rows": 1, "cols": 1, "dt": "d",
                                 "data": [value]}})

    with open(path) as handle:
        assert spelling in handle.read()


def test_a_long_data_array_wraps_where_opencv_wraps_it(yaml, tmp_path):
    """Seventy-two columns, and seven spaces of continuation.

    Measured from what FileStorage produces rather than read out of its
    source: a token ending at column 72 stays on the line and one that would
    end at 73 moves to the next.
    """
    path = tmp_path / "wrapped.yml"
    values = [10] + list(range(100, 120))
    yaml.write(str(path), {"A": {"rows": 1, "cols": len(values), "dt": "i",
                                 "data": values}})

    with open(path) as handle:
        lines = handle.read().splitlines()

    data = [line for line in lines if "data:" in line or line.startswith(" " * 7)]
    assert data[0] == "   data: [ 10, 100, 101, 102, 103, 104, 105, 106, " \
                      "107, 108, 109, 110,"
    assert len(data[0]) == 69
    assert data[1].startswith(" " * 7 + "111,")


# ----------------------------------------------------------------------------
# Corners no shipped fixture has
# ----------------------------------------------------------------------------

def test_scalars_keep_their_type(yaml, tmp_path):
    path = tmp_path / "scalars.yml"
    path.write_text("%YAML:1.0\n---\n"
                    "count: 5\n"
                    "ratio: 2.5\n"
                    "name: hello\n"
                    "quoted: \"12\"\n"
                    "spaced: hello world\n")

    document = yaml.read(str(path))

    assert document["count"] == 5 and isinstance(document["count"], int)
    assert document["ratio"] == 2.5 and isinstance(document["ratio"], float)
    assert document["name"] == "hello"
    assert document["quoted"] == "12", "quoting is what keeps a number a name"
    assert document["spaced"] == "hello world"


def test_a_nested_map_and_a_sequence(yaml, tmp_path):
    path = tmp_path / "nested.yml"
    path.write_text("%YAML:1.0\n---\n"
                    "outer:\n"
                    "   inner: 3\n"
                    "   list: [ 1, 2, 3 ]\n"
                    "flat: [ 1.5, 2.5 ]\n")

    document = yaml.read(str(path))

    assert document["outer"] == {"inner": 3, "list": [1, 2, 3]}
    assert document["flat"] == [1.5, 2.5]


def test_a_block_sequence_of_maps(yaml, tmp_path):
    """What `Model_SVM.xml`'s `<_>` entries are, in YAML syntax."""
    path = tmp_path / "sequence.yml"
    path.write_text("%YAML:1.0\n---\n"
                    "items:\n"
                    "   - id: 1\n"
                    "     name: first\n"
                    "   - id: 2\n"
                    "     name: second\n")

    assert yaml.read(str(path))["items"] == [
        {"id": 1, "name": "first"},
        {"id": 2, "name": "second"},
    ]


def test_a_matrix_reads_as_its_four_fields(yaml, tmp_path):
    """Not as a decoded array: `dt` says what the data is, and decoding here
    would throw it away."""
    path = tmp_path / "matrix.yml"
    path.write_text("%YAML:1.0\n---\n"
                    "M: !!opencv-matrix\n"
                    "   rows: 2\n"
                    "   cols: 2\n"
                    "   dt: d\n"
                    "   data: [ 1., 2., 3., 4. ]\n")

    assert yaml.read(str(path))["M"] == {
        "rows": 2, "cols": 2, "dt": "d", "data": [1.0, 2.0, 3.0, 4.0]}


def test_an_integer_matrix_keeps_its_integers(yaml, tmp_path):
    path = tmp_path / "ints.yml"
    yaml.write(str(path), {"I": {"rows": 1, "cols": 3, "dt": "i",
                                 "data": [1, 2, 3]}})

    with open(path) as handle:
        assert "data: [ 1, 2, 3 ]" in handle.read()

    assert yaml.read(str(path))["I"]["data"] == [1, 2, 3]


def test_a_missing_file_raises(yaml, tmp_path):
    with pytest.raises(ValueError):
        yaml.read(str(tmp_path / "nothing.yml"))


def test_xml_is_read(yaml):
    """`Model_SVM.xml` is the only XML VIAME parses itself."""
    source = os.path.join(REPO, documents()["model_svm_xml"])
    document = yaml.read(source)

    # Four classifiers, and `num_classes` says five: the hierarchy has one
    # more class than it has splits.
    assert document["num_classes"] == 5
    assert len(document["hier_part_classifier"]) == 4
    assert document["hier_part_classifier"][0]["svm_file"] == "speciesSVM_1.xml"


def test_xml_entities_are_unescaped(yaml, tmp_path):
    path = tmp_path / "escaped.xml"
    path.write_text('<?xml version="1.0"?>\n<opencv_storage>\n'
                    '<name>a &amp; b &lt;c&gt;</name>\n</opencv_storage>\n')

    assert yaml.read(str(path))["name"] == "a & b <c>"


def test_an_empty_document_is_an_empty_map(yaml, tmp_path):
    path = tmp_path / "empty.yml"
    path.write_text("%YAML:1.0\n---\n")

    assert yaml.read(str(path)) == {}
