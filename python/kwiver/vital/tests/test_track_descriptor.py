"""
ckwg +31
Copyright 2020 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Tests for Python interface to vital::track_descriptor

"""



import nose.tools as nt
from kwiver.vital.types import track_descriptor, uid, bounding_box, timestamp, descriptor
import numpy as np
# Tests for the inner class of TrackDescriptor
class TestVitalTrackDescriptorAndHistoryEntry(object):
    ##################### Tests on history_entry #####################
    def _create_history_entries(self):
        h1 = track_descriptor.HistoryEntry(
            timestamp.Timestamp(),
            bounding_box.BoundingBoxD(5, 10, 15, 20),
            bounding_box.BoundingBoxD(25, 30, 35, 40)
        )

        h2 = track_descriptor.HistoryEntry(
            timestamp.Timestamp(123, 1),
            bounding_box.BoundingBoxD(45, 50, 55, 60),
        )
        return (h1, h2)

    def test_create_history_entry(self):
        track_descriptor.HistoryEntry(
            timestamp.Timestamp(),
            bounding_box.BoundingBoxD(5, 10, 15, 20),
            bounding_box.BoundingBoxD(25, 30, 35, 40)
        )

        track_descriptor.HistoryEntry(
            timestamp.Timestamp(123, 1),
            bounding_box.BoundingBoxD(45, 50, 55, 60),
        )

    def test_get_timestamp(self):
        (h1, h2) = self._create_history_entries()

        # Test the first history entry timestamp
        # Can't use == because of definition for timestamp
        nt.assert_false(h1.get_timestamp().is_valid())

        # Can use == on second timestamp
        nt.assert_equals(h2.get_timestamp(), timestamp.Timestamp(123, 1))


    def test_get_image_location(self):
        (h1, h2) = self._create_history_entries()

        nt.assert_equals(h1.get_image_location(), bounding_box.BoundingBoxD(5, 10, 15, 20))
        nt.assert_equals(h2.get_image_location(), bounding_box.BoundingBoxD(45, 50, 55, 60))
    def test_get_world_location(self):
        (h1, h2) = self._create_history_entries()

        nt.assert_equals(h1.get_world_location(), bounding_box.BoundingBoxD(25, 30, 35, 40))
        nt.assert_equals(h2.get_world_location(), bounding_box.BoundingBoxD(0, 0, 0, 0))

    ##################### Tests on track_descriptor #####################

    def test_create_track_descriptors(self):
        td1 = track_descriptor.TrackDescriptor.create("test_type")
        td2 = track_descriptor.TrackDescriptor.create("")
        td1_copy = track_descriptor.TrackDescriptor.create(td1)
        td2_copy = track_descriptor.TrackDescriptor.create(td2)

    def _create_track_descriptors(self):
        td1 = track_descriptor.TrackDescriptor.create("test_type")
        td2 = track_descriptor.TrackDescriptor.create("")
        td1_copy = track_descriptor.TrackDescriptor.create(td1)
        td2_copy = track_descriptor.TrackDescriptor.create(td2)
        return [td1, td2, td1_copy, td2_copy]

    def test_set_and_get_type(self):
        tds = self._create_track_descriptors()
        default_types = ["test_type", "", "test_type", ""]
        for td, default_type in zip(tds, default_types):

            nt.assert_equals(td.type, default_type)

            td.type = "bar"
            nt.assert_equals(td.type, "bar")

            td.type = "baz"
            nt.assert_equals(td.type, "baz")

            td.type = "0"
            nt.assert_equals(td.type, "0")

            td.type = ""
            nt.assert_equals(td.type, "")

    def test_set_and_get_uid(self):
        tds = self._create_track_descriptors()
        for td in tds:
            # Check default
            nt.assert_equals(td.uid, uid.UID())

            uid1 = uid.UID("first")
            td.uid = uid1
            nt.assert_equals(td.uid, uid1)

            uid2 = uid.UID("second")
            td.uid = uid2
            nt.assert_equals(td.uid, uid2)

            uid_empty = uid.UID()
            td.uid = uid_empty
            nt.assert_equals(td.uid, uid_empty)

            td.uid = uid1
            nt.assert_equals(td.uid, uid1)

    def test_add_and_get_track_ids(self):
        tds = self._create_track_descriptors()
        for td in tds:
            td.add_track_id(14)
            nt.assert_equals(td.get_track_ids(), [14])

            td.add_track_id(0)
            nt.assert_equals(td.get_track_ids(), [14, 0])

            td.add_track_id(42)
            nt.assert_equals(td.get_track_ids(), [14, 0, 42])

            # Now test adding multiple track ids
            td.add_track_ids([100, 12345, 8274653])
            nt.assert_equals(td.get_track_ids(), [14, 0, 42, 100, 12345, 8274653])

    def _create_descriptors(self):
        dd1 = descriptor.new_descriptor(3, 'd')
        dd1[:] = 10

        dd2 = descriptor.new_descriptor(5, 'd')
        l = [5.1, -0.2, -5.63, 3.14, 2.71]
        for i in range(len(l)):
            dd2[i] = l[i]

        return [dd1, dd2]

    def test_set_and_get_descriptor(self):
        tds = self._create_track_descriptors()
        for td in tds:
            dds = self._create_descriptors()
            # Grab a copy to be sure that none of the descriptors
            # are changing when we set them
            dds_copy = self._create_descriptors()
            for dd, dd_copy in zip(dds, dds_copy):
                td.set_descriptor(dd)
                nt.assert_equals(td.descriptor_size(), len(dd.todoublearray()))
                dd_out = td.get_descriptor()
                np.testing.assert_array_almost_equal(dd_out.todoublearray(), dd_copy.todoublearray())

    # Just like in the c++ track_descriptor implementation,
    # we can get a reference to the internal descriptor
    # and make modifications through the reference
    def test_modify_descriptor_by_reference(self):
        tds = self._create_track_descriptors()
        for td in tds:
            # So we have some initial values
            td.resize_descriptor(3, 20)
            dd = td.get_descriptor()
            dd[0] = 99
            # establish that the first element changed in dd
            np.testing.assert_almost_equal(dd[0], 99)
            # establish that the first element changed in the
            # track descriptors internal descriptor member
            np.testing.assert_almost_equal(td[0], 99)
            # Check the rest didn't change
            np.testing.assert_almost_equal(td[1], 20)
            np.testing.assert_almost_equal(td[2], 20)

    def test_set_descriptor_wrong_type(self):
        tds = self._create_track_descriptors()
        for td in tds:
            nt.assert_raises(TypeError, td.set_descriptor, descriptor.new_descriptor(5, 'f'))


    def test_resize_descriptor_with_value(self):
        tds = self._create_track_descriptors()
        for td in tds:
            # adds the value of 20 to array 5 times
            td.resize_descriptor(5, 20)
            np.testing.assert_array_almost_equal(td.get_descriptor().todoublearray(), [20] * 5)
            nt.assert_equals(td.descriptor_size(), 5)

            td.resize_descriptor(2, 10)
            np.testing.assert_array_almost_equal(td.get_descriptor().todoublearray(), [10] * 2)
            nt.assert_equals(td.descriptor_size(), 2)

            td.resize_descriptor(8, -3.1415)
            np.testing.assert_array_almost_equal(td.get_descriptor().todoublearray(), [-3.1415] * 8)
            nt.assert_equals(td.descriptor_size(), 8)

            td.resize_descriptor(0, -3.1415)
            np.testing.assert_array_equal(td.get_descriptor().todoublearray(), [])
            nt.assert_equals(td.descriptor_size(), 0)


    def test_resize_descriptor_no_value(self):
        tds = self._create_track_descriptors()
        sizes = [5, 2, 29, 100, 0]
        for td in tds:
            for s in sizes:
                td.resize_descriptor(s)
                nt.assert_equals(td.descriptor_size(), s)



    def test_get_and_set_item(self):
        tds = self._create_track_descriptors()
        for td in tds:
            arr = np.random.uniform(low=0.0, high=1000, size = 10)
            td.resize_descriptor(10)
            for i in range(10):
                td[i] = arr[i]

            # Now check
            for i in range(10):
                np.testing.assert_almost_equal(td[i], arr[i],
                    err_msg = "value mismatch at index {}".format(i))

    # Note that 'at' only retrieves a value.
    # We can't use it to set a value at an idx, like in the c++ version
    def test_at(self):
        tds = self._create_track_descriptors()
        for td in tds:
            # Generates 10 random values between 0 and 1000
            arr = np.random.uniform(low=0.0, high=1000, size = 10)
            td.resize_descriptor(10)
            for i in range(10):
                td[i] = arr[i]

            # Now check
            for i in range(10):
                np.testing.assert_almost_equal(td.at(i), arr[i],
                    err_msg = "value mismatch at index {}".format(i))

    def test_bad_descriptor_array_access(self):
        tds = self._create_track_descriptors()
        sizes = [5, 0]
        for td in tds:
            for s in sizes:
                td.resize_descriptor(s, 20)
                # Check that we cant access index s
                with nt.assert_raises(IndexError):
                    dummy = td[s]

                with nt.assert_raises(IndexError):
                    dummy = td.at(s)

    def test_has_descriptor(self):
        tds = self._create_track_descriptors()
        for td in tds:
            nt.assert_false(td.has_descriptor())
            td.resize_descriptor(5)
            nt.assert_true(td.has_descriptor())

            td.resize_descriptor(0)
            nt.assert_false(td.has_descriptor())

    def test_set_and_get_history(self):
        tds = self._create_track_descriptors()
        histories = list(self._create_history_entries())
        histories_cpy = list(self._create_history_entries())
        for td in tds:
            nt.assert_equals(td.get_history(), [])
            td.set_history(histories)

            history_out = td.get_history()
            nt.assert_equals(len(history_out), 2)

            # Check that the first element is the same
            # == operator will return false since first timestamp is invalid
            nt.assert_false(history_out[0].get_timestamp().is_valid())
            nt.assert_equals(history_out[0].get_image_location(), histories_cpy[0].get_image_location())
            nt.assert_equals(history_out[0].get_world_location(), histories_cpy[0].get_world_location())

            # Check that the second element is the same
            nt.assert_equals(history_out[1], histories_cpy[1])


    def test_add_history_entry(self):
        tds = self._create_track_descriptors()
        histories = list(self._create_history_entries())
        histories_cpy = list(self._create_history_entries())

        for td in tds:
            nt.assert_equals(td.get_history(), [])

            td.add_history_entry(histories[0])
            history_out = td.get_history()
            nt.assert_false(history_out[0].get_timestamp().is_valid())
            nt.assert_equals(history_out[0].get_image_location(), histories_cpy[0].get_image_location())
            nt.assert_equals(history_out[0].get_world_location(), histories_cpy[0].get_world_location())

            td.add_history_entry(histories[1])
            history_out = td.get_history()
            # Check that the first is still correct
            nt.assert_false(history_out[0].get_timestamp().is_valid())
            nt.assert_equals(history_out[0].get_image_location(), histories_cpy[0].get_image_location())
            nt.assert_equals(history_out[0].get_world_location(), histories_cpy[0].get_world_location())

            # Can use == on the second
            nt.assert_equals(history_out[1], histories_cpy[1])
