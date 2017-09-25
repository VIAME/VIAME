"""
ckwg +31
Copyright 2015-2017 by Kitware, Inc.
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

Interface to KWIVER kwiver_process class.

"""
# -*- coding: utf-8 -*-


from sprokit.pipeline import process
from sprokit.pipeline import datum

import util.vital_type_converters as VTC

# import kwiver_process_utils


class KwiverProcess(process.PythonProcess):
    """This class represents a sprokit python process with traits extensions.
    This allows all derived processes to share a common set of types,
    ports and config entries.

    A common set of type and port traits is readily available from
    this class, and process can add additional traits as needed.

    Config traits are not that standardized, so no default set is provided.

    """

    class type_trait(object):
        """This class represents a single type trait. It binds together
        trait_name: name of this type specification used with other
        traits canonical type name: official system level type name
        string.  conv: function to convert data from boost::any to
        real type.

        The convert function takes in a "datum" and returns the correct type/

        These objects are indexed by _name

        """
        def __init__(self, tn, ctn, conv_in, conv_out):
            """
            :param tn: type trait name
            :param ctn: system level canonical type name string
            :param conv-in: converter function
            :param conv-out: converter function

            The converter function takes the source type and returns the
            destination type. If the source type is a PyCapsule, the
            return type is an opaque handle to the object. If the parameter
            is not a PyCapsule, then it is assumed to be an opaque handle to
            the appropriate type and a PyCapsule is returned.
            """
            self.name = tn
            self.canonical_name = ctn
            self.converter_in = conv_in
            self.converter_out = conv_out

    class port_trait(object):
        """
        This class represents a port trait. It binds together
        process level port name: name of the port on the process
        type trait: name of the type passed over this port
        description of port: a good description of this port
        """
        def __init__(self, nm, tt, descr):
            """
            nm: process level port name
            tt: type trait
            descr: port description
            """
            self.name = nm
            # reference to the real trait
            self.type_trait = tt
            self.description = descr

    class config_trait(object):
        """
        This class represents a config item. It binds together

        Need two CTORs
        """
        def __init__(self, name, key, default, descr):
            """
            :param name: name of the config trait
            :param key: name of the config entry
            :param default: default value for item (as string), optional
            :param descr: description of this config entry. Be explicit.
            """
            self.name = name
            self.key = key
            self.default = default
            self.description = descr

    # ----------------------------------------------------------
    # noinspection PyProtectedMember
    def __init__(self, conf):
        process.PythonProcess.__init__(self, conf)

        # establish the dictionaries for the declared traits.
        # indexed by their respective names
        #: :type: dict[str, KwiverProcess.type_trait]
        self._type_trait_set = dict()
        #: :type: dict[str, KwiverProcess.port_trait]
        self._port_trait_set = dict()
        #: :type: dict[str, KwiverProcess.config_trait]
        self._config_trait_set = dict()

        #
        # default set of kwiver/vital type traits
        #    These definitions must be the same as those in kwiver_type_traits.h
        #
        # Converter functions are needed for most vital data types
        # (e.g. kwiver:image). Types that resolve to basic types (e.g. float,
        # string) have converters in the sprokit python support code and will be
        # used if no converter function is specified.
        #
        # If there is no converter in sprokit and a conversion function is not
        # specified as part of the trait, fire will rain from above.
        #
        self.add_type_trait("timestamp", "kwiver:timestamp")
        self.add_type_trait("gsd", "kwiver:gsd")
        self.add_type_trait("image", "kwiver:image",
                            VTC._convert_image_container_in,
                            VTC._convert_image_container_out)
        self.add_type_trait("mask", "kwiver:image",
                            VTC._convert_image_container_in,
                            VTC._convert_image_container_out)
        self.add_type_trait("feature_set", "kwiver:feature_set")
        self.add_type_trait("descriptor_set", "kwiver:descriptor_set",
                            VTC.convert_descriptor_set_in,
                            VTC.convert_descriptor_set_out)
        self.add_type_trait("detected_object_set", "kwiver:detected_object_set",
                            VTC._convert_detected_object_set_in,
                            VTC._convert_detected_object_set_out)
        self.add_type_trait("track_set", "kwiver:track_set",
                            VTC._convert_track_set_handle)
        self.add_type_trait("feature_track_set", "kwiver:feature_track_set",
                            VTC._convert_track_set_handle)
        self.add_type_trait("object_track_set", "kwiver:object_track_set",
                            VTC._convert_track_set_handle)
        self.add_type_trait("homography_src_to_ref", "kwiver:s2r_homography")
        self.add_type_trait("homography_ref_to_src", "kwiver:r2s_homography")
        self.add_type_trait("image_file_name", "kwiver:image_file_name")
        self.add_type_trait("video_file_name", "kwiver:video_file_name")

        self.add_type_trait("double_vector", "kwiver:d_vector",
                            VTC._convert_double_vector_in,
                            VTC._convert_double_vector_out)
        self.add_type_trait("string_vector", "kwiver:string_vector",
                            VTC.convert_string_vector_in,
                            VTC.convert_string_vector_out)

        #                   port-name    type-trait-name    description
        self.add_port_trait("timestamp", "timestamp",
                            "Timestamp for input image")
        self.add_port_trait("image", "image", "Single frame input image")
        self.add_port_trait("mask", "mask", "Imput mask image")
        self.add_port_trait("feature_set", "feature_set",
                            "Set of detected features")
        self.add_port_trait("descriptor_set", "descriptor_set",
                            "Set of feature descriptors")
        self.add_port_trait("detected_object_set", "detected_object_set",
                            "Set of object detections")
        self.add_port_trait("track_set", "track_set",
                            "Set of arbitrary tracks")
        self.add_port_trait("feature_track_set", "feature_track_set",
                            "Set of feature tracks")
        self.add_port_trait("object_track_set", "object_track_set",
                            "Set of object tracks")
        self.add_port_trait("homography_src_to_ref", "homography_src_to_ref",
                            "Source image to ref image homography.")
        self.add_port_trait("image_file_name", "image_file_name",
                            "Name of an image file. Usually a single frame of "
                            "a video.")
        self.add_port_trait("video_file_name", "video_file_name",
                            "Name of video file.")

    def add_type_trait(self, ttn, tn, conv_in=None, conv_out=None):
        """
        Create a type trait and add to this process so it can be used.

        Parameters:
        :param ttn: type trait name (type trait must be previously created)
        :param tn: canonical type name for this application
        :param conv_in: converter function (optional)
        :param conv_out: converter function (optional)

        """
        self._type_trait_set[ttn] = self.type_trait(ttn, tn, conv_in, conv_out)

    def add_port_trait(self, nm, ttn, descr):
        """
        Create a port trait so it can be used with this process.

        Parameters:
        :param nm: trait name. Also used as port name.
        :param ttn: type trait name (type trait must be previously created)
        :param descr: description of port

        """
        # check to see if tn is in set below
        tt = self._type_trait_set.get(ttn)
        if tt is None:
            raise ValueError('type trait name \"%s\" not registered' % ttn)
        self._port_trait_set[nm] = self.port_trait(nm, tt, descr)

    def add_config_trait(self, name, key, default, descr):
        """
        Create a config trait and add to this process. Once a config trait is
        created it can be used to declare and access a config entry.

        :param name: trait name
        :param key: config key string
        :param default: default value string
        :param descr: description

        """
        self._config_trait_set[name] = self.config_trait(name, key, default,
                                                         descr)

    # ----------------------------------------------------------
    def declare_input_port_using_trait(self, ptn, flag):
        """
        Declare a port on the specified process using the pre-defined trait.

        :param ptn: port trait name
        :param flag: required/optional flags

        """
        port_trait = self._port_trait_set.get(ptn, None)
        if port_trait is None:
            raise ValueError('port trait name \"%s\" not registered' % ptn)

        self.declare_input_port(port_trait.name,
                                port_trait.type_trait.canonical_name,
                                flag,
                                port_trait.description)

    def declare_output_port_using_trait(self, ptn, flag):
        """
        Declare a port on the specified process using the pre-defined trait.

        :param ptn: port trait name
        :param flag: required/optional flags

        """
        port_trait = self._port_trait_set.get(ptn, None)
        if port_trait is None:
            raise ValueError('port trait name \"%s\" not registered' % ptn)

        self.declare_output_port(port_trait.name,
                                 port_trait.type_trait.canonical_name,
                                 flag,
                                 port_trait.description)

    def grab_input_using_trait(self, ptn):
        """
        Get value from port using traits.

        The datum is pulled from port and if the type-trait associated with
        this port has a converter function, the data is passed to that
        function. If there is no converter regietered, then the raw datum is
        returned.

        This call is used to return managed types such as image_container, 
        track_set.

        The raw datum contains the port data and other metadata.
        """
        pt = self._port_trait_set.get(ptn, None)
        if pt is None:
            raise ValueError('port trait name \"%s\" not registered' % ptn)

        pipeline_datum = self.grab_datum_from_port(pt.name)
        tt = pt.type_trait
        if tt.converter_in is not None:
            data = tt.converter_in(pipeline_datum.get_datum_ptr())
            return data

        return pipeline_datum

    def grab_value_using_trait(self, ptn):
        """
        Get sprokit datum value from port. The caller is responsible for
        converting the datum to an acceptable value.

        The value from the port datum is converted into a python type if
        there is a converter registered in sprokit. This is usually limited to
        fundamental types, such as int, double, bool, string, char
        """
        pt = self._port_trait_set.get(ptn, None)
        if pt is None:
            raise ValueError('port trait name \"%s\" not registered' % ptn)

        return self.grab_value_from_port(pt.name)

    def declare_config_using_trait(self, name):
        """
        Declare a process config entry from the named trait.
        An exception will be thrown if the config trait has not been registered
        with the process.
        """
        ct = self._config_trait_set.get(name, None)
        if ct is None:
            raise ValueError('config trait name \"%s\" not registered' % name)

        process.PythonProcess.declare_configuration_key(self, ct.key,
                                                        ct.default,
                                                        ct.description)

    def config_value_using_trait(self, name):
        """
        Get value from config using trait.
        An exception will be thrown if the config trait has not been registered
        with the process.

        :param name: Name of the configuration trait.

        """
        ct = self._config_trait_set.get(name, None)
        if ct is None:
            raise ValueError('config trait name \"%s\" not registered' % name)

        return self.config_value(ct.name)

    def push_to_port_using_trait(self, ptn, val):
        """
        Push value to port using trait.

        :param ptn: port trait name
        :param val: value to put on port

        If the trait has a converter function, the supplied value will be 
        converted by that function to a datum which will be pushed to the port.

        If no converter is associated with the trait, the raw value supplied 
        will be pushed to the port. If the value is already a datum, then all is 
        well. If it is some other data type, such as a fundamental type, it will 
        be automatically be converted to a datum.

        """
        pt = self._port_trait_set.get(ptn, None)
        if pt is None:
            raise ValueError('port trait name \"%s\" not registered' % ptn)

        tt = pt.type_trait
        if tt.converter_out is not None:
            # convert handle to PyCapsule around datum ptr
            cap = tt.converter_out(val)
            # convert to datum_t
            dat = datum.datum_from_capsule(cap)
            self.push_datum_to_port(pt.name, dat)
        else:
            # no registered converter - hope for the best
            self.push_datum_to_port(pt.name, val)

    def push_datum_to_port_using_trait(self, ptn, val):
        """
        Push datum to port using trait.

        The datum has already been formed, so it is pushed directly.

        :param ptn: port trait name
        :param val: datum to push to port

        """
        pt = self._port_trait_set.get(ptn, None)
        if pt is None:
            raise ValueError('port trait name \"%s\" not registered' % ptn)

        self.push_datum_to_port(pt.name, val)
