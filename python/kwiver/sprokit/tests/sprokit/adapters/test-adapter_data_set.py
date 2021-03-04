#!/usr/bin/env python
# ckwg +28
# Copyright 2020 by Kitware, Inc.
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

from kwiver.sprokit.pipeline import datum

def test_import():
    try:
        import kwiver.sprokit.adapters.adapter_data_set
    except:
        test_error("Failed to import the adapter_data_set module")


def test_create():
    from kwiver.sprokit.adapters import adapter_data_set

    adapter_data_set.AdapterDataSet.create()
    adapter_data_set.AdapterDataSet.create(adapter_data_set.DataSetType.data)
    adapter_data_set.AdapterDataSet.create(adapter_data_set.DataSetType.end_of_input)


def check_type():
    from kwiver.sprokit.adapters import adapter_data_set

    ads = (
        adapter_data_set.AdapterDataSet.create()
    )  # Check constructor with default argument
    ads_data = adapter_data_set.AdapterDataSet.create(adapter_data_set.DataSetType.data)
    ads_eoi = adapter_data_set.AdapterDataSet.create(
        adapter_data_set.DataSetType.end_of_input
    )

    if ads_def.type() != adapter_data_set.DataSetType.data:
        test_error("adapter_data_set type mismatch: constructor with default arg")

    if ads_data.type() != adapter_data_set.DataSetType.data:
        test_error("adapter_data_set type mismatch: constructor with data arg")

    if ads_eoi.type() != adapter_data_set.DataSetType.end_of_input:
        test_error("adapter_data_set type mismatch: constructor with end_of_input arg")


def test_enums():
    from kwiver.sprokit.adapters import adapter_data_set

    if int(adapter_data_set.DataSetType.data) != 1:
        test_error("adapter_data_set enum value mismatch: data")

    if int(adapter_data_set.DataSetType.end_of_input) != 2:
        test_error("adapter_data_set enum value mismatch: end_of_input")


def test_is_end_of_data():
    from kwiver.sprokit.adapters import adapter_data_set

    ads_def = adapter_data_set.AdapterDataSet.create()  # test default argument
    ads_data = adapter_data_set.AdapterDataSet.create(adapter_data_set.DataSetType.data)
    ads_eoi = adapter_data_set.AdapterDataSet.create(
        adapter_data_set.DataSetType.end_of_input
    )

    if ads_def.is_end_of_data():
        test_error(
            'adapter data set of type "data" is empty: constructor with default arg'
        )

    if ads_data.is_end_of_data():
        test_error(
            'adapter data set of type "data" is empty: constructor with data arg'
        )

    if not ads_eoi.is_end_of_data():
        test_error('adapter_data_set of type "end_of_input" is not empty')


def check_same_type(retrieved_val, val, portname):
    from kwiver.sprokit.adapters import adapter_data_set

    if isinstance(val, datum.Datum):
        val = val.get_datum()
    if not type(retrieved_val) is type(val):
        msg = "Retrieved value of type: {} at port {}. Expected type: {}"
        msg = msg.format(type(retrieved_val), portname, type(val))
        test_error(msg)


# adds and retrieves val to/from the
# adapter_data_set instance 3 times
# once with the add/get fxn specified,
# once with the add/get function that automatically casts,
# once with the index operator
def add_get_helper(
    instance, instance_add_fxn, instance_get_fxn, val, data_type_str,
):
    from kwiver.sprokit.adapters import adapter_data_set

    # First the type specific add/get fxns
    portname = data_type_str + "_port"
    instance_add_fxn(portname, val)
    retrieved_val = instance_get_fxn(portname)  # Throws if port not found
    check_same_type(retrieved_val, val, portname)

    # Next the automatic type handling add/get fxns
    # First add_value and get_port_data
    portname = "py_" + portname
    instance.add_value(portname, val)
    retrieved_val = instance.get_port_data(portname)
    check_same_type(retrieved_val, val, portname)

    # Now __getitem__ and __setitem__
    portname += "2"
    instance[portname] = val
    retrieved_val = instance[portname]
    check_same_type(retrieved_val, val, portname)


def overwrite_helper(
    instance_add_fxn, instance_get_fxn, val, new_data_type_str, portname
):
    from kwiver.sprokit.adapters import adapter_data_set

    instance_add_fxn(portname, val)
    try:
        retrieved_val = instance_get_fxn(portname)
    except RuntimeError:
        test_error(
            "Failed to get object of type {} after attempting overwrite".format(
                new_data_type_str
            )
        )
    else:
        if isinstance(val, datum.Datum):
            val = val.get_datum()
        if retrieved_val != val:
            test_error(
                "Retrieved incorrect value after overwriting with {}".format(
                    new_data_type_str
                )
            )


def test_empty():
    from kwiver.sprokit.adapters import adapter_data_set

    if not adapter_data_set.AdapterDataSet.create().empty():
        test_error("fresh data adapter_data_set instance is not empty")


# First check a python datum object
def test_add_get_datum():
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()
    add_get_helper(ads, ads.add_datum, ads.get_port_data, datum.new("d1"), "datum")


# Next some basic types
def test_add_get_basic_types():
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()
    add_get_helper(ads, ads._add_int, ads._get_port_data_int, 10, "int")
    add_get_helper(ads, ads._add_float, ads._get_port_data_float, 0.5, "float")
    add_get_helper(ads, ads._add_string, ads._get_port_data_string, "str1", "string")


# Next some kwiver vital types that are handled with pointers
def test_add_get_vital_types_by_ptr():
    from kwiver.sprokit.adapters import adapter_data_set
    from kwiver.vital import types as kvt

    ads = adapter_data_set.AdapterDataSet.create()
    add_get_helper(
        ads,
        ads._add_image_container,
        ads._get_port_data_image_container,
        kvt.ImageContainer(kvt.Image()),
        "image_container",
    )
    add_get_helper(
        ads,
        ads._add_descriptor_set,
        ads._get_port_data_descriptor_set,
        kvt.DescriptorSet(),
        "descriptor_set",
    )
    add_get_helper(
        ads,
        ads._add_detected_object_set,
        ads._get_port_data_detected_object_set,
        kvt.DetectedObjectSet(),
        "detected_object_set",
    )
    add_get_helper(
        ads,
        ads._add_track_set,
        ads._get_port_data_track_set,
        kvt.TrackSet(),
        "track_set",
    )
    add_get_helper(
        ads,
        ads._add_object_track_set,
        ads._get_port_data_object_track_set,
        kvt.ObjectTrackSet(),
        "object_track_set",
    )


# Next some bound native C++ types
def test_add_get_cpp_types():
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()
    add_get_helper(
        ads,
        ads._add_double_vector,
        ads._get_port_data_double_vector,
        adapter_data_set.VectorDouble([3.14, 4.14]),
        "double_vector",
    )
    add_get_helper(
        ads,
        ads._add_string_vector,
        ads._get_port_data_string_vector,
        adapter_data_set.VectorString(["s00", "s01"]),
        "string_vector",
    )
    add_get_helper(
        ads,
        ads._add_uchar_vector,
        ads._get_port_data_uchar_vector,
        adapter_data_set.VectorUChar([100, 101]),
        "uchar_vector",
    )


# Now try creating datums of these bound types
def test_add_get_cpp_types_with_datum():
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()
    add_get_helper(
        ads,
        ads.add_datum,
        ads._get_port_data_double_vector,
        datum.new_double_vector(datum.VectorDouble([6.3, 8.9])),
        "datum_double_vector",
    )
    add_get_helper(
        ads,
        ads.add_datum,
        ads._get_port_data_string_vector,
        datum.new_string_vector(datum.VectorString(["foo", "bar"])),
        "datum_string_vector",
    )
    add_get_helper(
        ads,
        ads.add_datum,
        ads._get_port_data_uchar_vector,
        datum.new_uchar_vector(datum.VectorUChar([102, 103])),
        "datum_uchar_vector",
    )


# Next kwiver vital types
def test_add_get_vital_types():
    from kwiver.vital import types as kvt
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()
    add_get_helper(
        ads,
        ads._add_bounding_box,
        ads._get_port_data_bounding_box,
        kvt.BoundingBoxD(1, 1, 2, 2),
        "bounding_box",
    )
    add_get_helper(
        ads,
        ads._add_timestamp,
        ads._get_port_data_timestamp,
        kvt.Timestamp(),
        "timestamp",
    )
    add_get_helper(
        ads,
        ads._add_f2f_homography,
        ads._get_port_data_f2f_homography,
        kvt.F2FHomography(1),
        "f2f_homography",
    )


# Now test overwriting
def test_overwrite():
    from kwiver.vital import types as kvt
    from kwiver.sprokit.adapters import adapter_data_set

    OVERWRITE_PORT = "test_overwrite_port"
    ads = adapter_data_set.AdapterDataSet.create()

    # Overwriting with same datum
    ads.add_datum(OVERWRITE_PORT, datum.new("d2"))
    overwrite_helper(
        ads.add_datum,
        ads.get_port_data,
        datum.new("d3"),
        "datum_string",
        OVERWRITE_PORT,
    )

    # Overwriting with completely different types
    overwrite_helper(
        ads.add_datum, ads.get_port_data, datum.new(12), "datum_int", OVERWRITE_PORT
    )
    overwrite_helper(
        ads._add_string_vector,
        ads._get_port_data_string_vector,
        adapter_data_set.VectorString(["baz", "qux"]),
        "string_vector",
        OVERWRITE_PORT,
    )
    overwrite_helper(
        ads._add_timestamp,
        ads._get_port_data_timestamp,
        kvt.Timestamp(100, 10),
        "timestamp",
        OVERWRITE_PORT,
    )
    overwrite_helper(ads.add_value, ads.get_port_data, 15, "int", OVERWRITE_PORT)
    overwrite_helper(
        ads._add_double_vector,
        ads._get_port_data_double_vector,
        adapter_data_set.VectorDouble([4, 8]),
        "double_vector",
        OVERWRITE_PORT,
    )


# Want to make sure data inside a datum created with the automatic
# conversion constructor can be retrieved with a type specific getter, and
# vice versa
def test_mix_add_and_get():
    from kwiver.sprokit.adapters import adapter_data_set
    from kwiver.vital import types as kvt

    ads = adapter_data_set.AdapterDataSet.create()

    # Try adding with generic adder first, retrieving with
    # type specific get function
    ads["string_port"] = "string_value"
    check_same_type(
        ads._get_port_data_string("string_port"), "string_value", "string_port"
    )

    ads["timestamp_port"] = kvt.Timestamp(1000000000, 10)
    check_same_type(
        ads._get_port_data_timestamp("timestamp_port"),
        kvt.Timestamp(1000000000, 10),
        "timestamp_port",
    )

    ads["vector_string_port"] = adapter_data_set.VectorString(["element1", "element2"])
    check_same_type(
        ads._get_port_data_string_vector("vector_string_port"),
        adapter_data_set.VectorString(["element1", "element2"]),
        "vector_string_port",
    )

    # Now try the opposite
    ads._add_string("string_port", "string_value")
    check_same_type(ads["string_port"], "string_value", "string_port")

    ads._add_timestamp("timestamp_port", kvt.Timestamp(1000000000, 10))
    check_same_type(
        ads["timestamp_port"], kvt.Timestamp(1000000000, 10), "timestamp_port"
    )

    ads._add_string_vector(
        "vector_string_port", adapter_data_set.VectorString(["element1", "element2"])
    )
    check_same_type(
        ads["vector_string_port"],
        adapter_data_set.VectorString(["element1", "element2"]),
        "vector_string_port",
    )


# Make sure that None isn't acceptable, even for pointers
def test_add_none():
    from kwiver.sprokit.adapters import adapter_data_set
    from kwiver.vital import types as kvt

    ads = adapter_data_set.AdapterDataSet.create()

    expect_exception(
        "attempting to store None as a string vector",
        TypeError,
        ads._add_string_vector,
        "none_vector_string_port",
        None,
    )

    expect_exception(
        "attempting to store None as a track set",
        TypeError,
        ads._add_track_set,
        "none_track_set_port",
        None,
    )

    expect_exception(
        "attempting to store None as a timestamp",
        TypeError,
        ads._add_timestamp,
        "none_timestamp_port",
        None,
    )

    # Should also fail for the automatic type conversion
    expect_exception(
        "attempting to store none through automatic conversion",
        TypeError,
        ads.add_value,
        "none_port",
        None,
    )


def _create_ads():
    from kwiver.vital import types as kvt
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()

    # Construct a few elements
    ads["string_port"] = "string_value"
    ads["timestamp_port"] = kvt.Timestamp(1000000000, 10)
    ads["vector_string_port"] = adapter_data_set.VectorString(["element1", "element2"])

    return ads


def test_iter():
    from kwiver.vital import types as kvt
    from kwiver.sprokit.adapters import adapter_data_set

    ads = _create_ads()

    for port, dat in ads:
        if port == "string_port":
            if dat.get_datum() != "string_value":
                test_error("Didn't retrieve correct string value on first iteration")
        elif port == "timestamp_port":
            if dat.get_datum() != kvt.Timestamp(1000000000, 10):
                test_error(
                    "Didn't retrieve correct timestamp value on second iteration"
                )
        elif port == "vector_string_port":
            if dat.get_datum() != datum.VectorString(["element1", "element2"]):
                test_error("Didn't retrieve correct string vector on third iteration")
        else:
            test_error("unknown port: {}".format(port))


def check_formatting_fxn(exp, act, fxn_name):
    if not act == exp:
        test_error("Expected {} to return '{}'. Got '{}'".format(fxn_name, exp, act))

    print(act)


def test_nice():
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()
    check_formatting_fxn("size=0", ads.__nice__(), "__nice__")

    ads = _create_ads()
    check_formatting_fxn("size=3", ads.__nice__(), "__nice__")


def test_repr():
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()
    exp = "<AdapterDataSet(size=0) at {}>".format(hex(id(ads)))
    check_formatting_fxn(exp, ads.__repr__(), "__repr__")

    ads = _create_ads()
    exp = "<AdapterDataSet(size=3) at {}>".format(hex(id(ads)))
    check_formatting_fxn(exp, ads.__repr__(), "__repr__")


def test_str():
    from kwiver.sprokit.adapters import adapter_data_set

    # Formatted string we'll fill in for each ads below
    exp_stem = "<AdapterDataSet(size={})>\n\t{{{}}}"

    ads = adapter_data_set.AdapterDataSet.create()
    check_formatting_fxn(exp_stem.format(0, ""), ads.__str__(), "__str__")

    ads = _create_ads()
    # This one actually has content, so we'll have to manually derive it
    content = ""
    content += "string_port: " + str(ads["string_port"])
    content += ", timestamp_port: " + str(ads["timestamp_port"])
    content += ", vector_string_port: " + str(ads["vector_string_port"])
    check_formatting_fxn(exp_stem.format(3, content), ads.__str__(), "__str__")


def test_len():
    from kwiver.sprokit.adapters import adapter_data_set

    ads = adapter_data_set.AdapterDataSet.create()

    # Check initial
    if len(ads) != 0:
        test_error("adapter_data_set with 0 values returned size {}".format(len(ads)))

    ads = _create_ads()

    if len(ads) != 3:
        test_error("adapter_data_set with 3 values returned size {}".format(len(ads)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        test_error("Expected two arguments")
        sys.exit(1)

    testname = sys.argv[1]

    run_test(testname, find_tests(locals()))
