#!/usr/bin/env python
#ckwg +28
# Copyright 2011-2020 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from kwiver.sprokit.util.test import expect_exception, find_tests, run_test, test_error

def test_import():
    try:
        import kwiver.sprokit.pipeline.datum
    except:
        test_error("Failed to import the datum module")


def test_new():
    from kwiver.sprokit.pipeline import datum

    d = datum.new('test_datum')

    if not d.type() == datum.DatumType.data:
        test_error("Datum type mismatch")

    if len(d.get_error()):
        test_error("A data datum has an error string")

    p = d.get_datum()

    if p is None:
        test_error("A data datum has None as its data")


def test_empty():
    from  kwiver.sprokit.pipeline import datum

    d = datum.empty()

    if not d.type() == datum.DatumType.empty:
        test_error("Datum type mismatch")

    if len(d.get_error()):
        test_error("An empty datum has an error string")

    p = d.get_datum()

    if p is not None:
        test_error("An empty datum does not have None as its data")


def test_flush():
    from  kwiver.sprokit.pipeline import datum

    d = datum.flush()

    if not d.type() == datum.DatumType.flush:
        test_error("Datum type mismatch")

    if len(d.get_error()):
        test_error("A flush datum has an error string")

    p = d.get_datum()

    if p is not None:
        test_error("A flush datum does not have None as its data")


def test_complete():
    from  kwiver.sprokit.pipeline import datum

    d = datum.complete()

    if not d.type() == datum.DatumType.complete:
        test_error("Datum type mismatch")

    if len(d.get_error()):
        test_error("A complete datum has an error string")

    p = d.get_datum()

    if p is not None:
        test_error("A complete datum does not have None as its data")


def test_error_():
    from  kwiver.sprokit.pipeline import datum

    err = 'An error'

    d = datum.error(err)

    if not d.type() == datum.DatumType.error:
        test_error("Datum type mismatch")

    if not d.get_error() == err:
        test_error("An error datum did not keep the message")

    p = d.get_datum()

    if p is not None:
        test_error("An error datum does not have None as its data")

def check_same_type(retrieved_val, val):
    if not type(retrieved_val) is type(val):
        msg = "Retrieved value of type: {}. Expected type: {}"
        msg = msg.format(type(retrieved_val), type(val))
        test_error(msg)

# Check the automatic type conversion done by new() and get_datum()
def check_automatic_conversion(val):
    from kwiver.sprokit.pipeline import datum

    datum_inst = datum.new(val)
    retrieved_val = datum_inst.get_datum()
    check_same_type(retrieved_val, val)

# Next some basic types
def test_add_get_basic_types():
    from kwiver.sprokit.pipeline import datum

    # Try the typed constructor/get fxns first
    datum_inst = datum.new_int(10)
    retrieved_val = datum_inst.get_int()
    check_same_type(retrieved_val, 10)

    datum_inst = datum.new_float(0.5)
    retrieved_val = datum_inst.get_float()
    check_same_type(retrieved_val, 0.5)

    datum_inst = datum.new_string("str1")
    retrieved_val = datum_inst.get_string()
    check_same_type(retrieved_val, "str1")

    # Now the ones with automatic conversion
    check_automatic_conversion(10)
    check_automatic_conversion(0.5)
    check_automatic_conversion("str1")

# Next some kwiver vital types that are handled with pointers
def test_add_get_vital_types_by_ptr():
    from kwiver.sprokit.pipeline import datum
    from kwiver.vital import types as kvt

    datum_inst = datum.new_image_container(kvt.ImageContainer(kvt.Image()))
    retrieved_val = datum_inst.get_image_container()
    check_same_type(retrieved_val, kvt.ImageContainer(kvt.Image()))

    datum_inst = datum.new_descriptor_set(kvt.DescriptorSet())
    retrieved_val = datum_inst.get_descriptor_set()
    check_same_type(retrieved_val, kvt.DescriptorSet())

    datum_inst = datum.new_detected_object_set(kvt.DetectedObjectSet())
    retrieved_val = datum_inst.get_detected_object_set()
    check_same_type(retrieved_val, kvt.DetectedObjectSet())

    datum_inst = datum.new_track_set(kvt.TrackSet())
    retrieved_val = datum_inst.get_track_set()
    check_same_type(retrieved_val, kvt.TrackSet())

    datum_inst = datum.new_object_track_set(kvt.ObjectTrackSet())
    retrieved_val = datum_inst.get_object_track_set()
    check_same_type(retrieved_val, kvt.ObjectTrackSet())

    check_automatic_conversion(kvt.ImageContainer(kvt.Image()))
    check_automatic_conversion(kvt.DescriptorSet())
    check_automatic_conversion(kvt.DetectedObjectSet())
    check_automatic_conversion(kvt.TrackSet())
    check_automatic_conversion(kvt.ObjectTrackSet())

# Next some bound native C++ types
def test_add_get_cpp_types():
    from kwiver.sprokit.pipeline import datum

    datum_inst = datum.new_double_vector(datum.VectorDouble([3.14, 4.14]))
    retrieved_val = datum_inst.get_double_vector()
    check_same_type(retrieved_val, datum.VectorDouble([3.14, 4.14]))

    datum_inst = datum.new_string_vector(datum.VectorString(["s00", "s01"]))
    retrieved_val = datum_inst.get_string_vector()
    check_same_type(retrieved_val, datum.VectorString(["s00", "s01"]))

    datum_inst = datum.new_uchar_vector(datum.VectorUChar([100, 101]))
    retrieved_val = datum_inst.get_uchar_vector()
    check_same_type(retrieved_val, datum.VectorUChar([100, 101]))

    check_automatic_conversion(datum.VectorDouble([3.14, 4.14]))
    check_automatic_conversion(datum.VectorString(["s00", "s01"]))
    check_automatic_conversion(datum.VectorUChar([100, 101]))

# Next kwiver vital types
def test_add_get_vital_types():
    from kwiver.sprokit.pipeline import datum
    from kwiver.vital import types as kvt

    datum_inst = datum.new_bounding_box(kvt.BoundingBoxD(1, 1, 2, 2))
    retrieved_val = datum_inst.get_bounding_box()
    check_same_type(retrieved_val, kvt.BoundingBoxD(1, 1, 2, 2))

    datum_inst = datum.new_timestamp(kvt.Timestamp())
    retrieved_val = datum_inst.get_timestamp()
    check_same_type(retrieved_val, kvt.Timestamp())

    datum_inst = datum.new_f2f_homography(kvt.F2FHomography(1))
    retrieved_val = datum_inst.get_f2f_homography()
    check_same_type(retrieved_val, kvt.F2FHomography(1))

    check_automatic_conversion(kvt.BoundingBoxD(1, 1, 2, 2))
    check_automatic_conversion(kvt.Timestamp())
    check_automatic_conversion(kvt.F2FHomography(1))

# Want to make sure data inside a datum created with the automatic
# conversion constructor can be retrieved with a type specific getter, and
# vice versa
def test_mix_new_and_get():
    from kwiver.sprokit.pipeline import datum
    from kwiver.vital import types as kvt

    # Try creating with generic constructor first, retrieving with
    # type specific get function
    datum_inst = datum.new("string_value")
    check_same_type(datum_inst.get_string(), "string_value")

    datum_inst = datum.new(kvt.Timestamp(1000000000, 10))
    check_same_type(datum_inst.get_timestamp(), kvt.Timestamp(1000000000, 10))

    datum_inst = datum.new(datum.VectorString(["element1", "element2"]))
    check_same_type(datum_inst.get_string_vector(), datum.VectorString(["element1", "element2"]))

    # Now try the opposite
    datum_inst = datum.new_string("string_value")
    check_same_type(datum_inst.get_datum(), "string_value")

    datum_inst = datum.new_timestamp(kvt.Timestamp(1000000000, 10))
    check_same_type(datum_inst.get_datum(), kvt.Timestamp(1000000000, 10))

    datum_inst = datum.new_string_vector(datum.VectorString(["element1", "element2"]))
    check_same_type(datum_inst.get_datum(), datum.VectorString(["element1", "element2"]))

# Make sure that None isn't acceptable, even for pointers
def test_new_with_none():
    from kwiver.sprokit.pipeline import datum
    from kwiver.vital import types as kvt

    expect_exception(
        "attempting to store None as a string vector",
        TypeError,
        datum.new_string_vector,
        None,
    )

    expect_exception(
        "attempting to store None as a track_set",
        TypeError,
        datum.new_track_set,
        None,
    )

    expect_exception(
        "attempting to store None as a timestamp",
        TypeError,
        datum.new_timestamp,
        None,
    )

    # Should also fail for the automatic type conversion
    expect_exception(
        "attempting to store none through automatic conversion",
        TypeError,
        datum.new,
        None,
    )

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        test_error("Expected two arguments")
        sys.exit(1)

    testname = sys.argv[1]


    run_test(testname, find_tests(locals()))
