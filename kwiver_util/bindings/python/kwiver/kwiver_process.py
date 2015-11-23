"""
ckwg +31
Copyright 2015 by Kitware, Inc.
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

import util.vital_type_converters as VTC

# import kwiver_process_utils

class KwiverProcess(process.PythonProcess):
    """
    This class represents a sprokit python process with traits extensions.
    This allows all derived processes to share a common set of types,
    ports and config entries. A proces scan add additional traits as needed.
    """

    class type_trait(object):
        """
        This class represents a single type trait. It binds together
        trait_name: name of this type specification used with other traits
        canonical type name: official system level type name string.
        conv: function to convert data from boost::any to real type.

        The convert function takes in a "datum" and returns the correct type/

        These objects are indexed by _name
        """
        def __init__(self, tn, ctn, conv):
            """
            tn: type trait name
            ctn: system level canonical type name string
            """
            self.name = tn
            self.canonical_name = ctn
            self.converter = conv


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
            self.type_trait = tt # reference to the real trait
            self.description = descr


    class config_trait(object):
        """
        This class represents a config item. It binds together
        name: name of the config trait
        key: name of the config entry
        default: default value for item (as string), optional
        descr: description of this config entry. Be explicit.

        Need two CTORs
        """
        def __init__( self, name, key, default, descr):
            """
            key: config key string
            default: default value, could be ""
            descr: description of config entry
            """
            self.key = key
            self.default = default
            self.description = descr


    # ----------------------------------------------------------
    def __init__(self, conf):
        process.PythonProcess.__init__(self, conf)

        # establish the dictionaries for the declared traits.
        # indexed by their respective names
        self._type_trait_set = dict()
        self._port_trait_set = dict()
        self._config_trait_set = dict()

        #
        # default set of kwiver/vital type traits
        #    These definitions must be the same as those in kwiver_type_traits.h
        #
        #                   trait name, system-level-type,   converter function
        self.add_type_trait("timestamp", "kwiver::timestamp")
        self.add_type_trait("gsd", "kwiver:gsd")
        self.add_type_trait("image", "kwiver:image", VTC._convert_image_container_sptr )
        self.add_type_trait("mask", "kwiver:image", VTC._convert_image_container_sptr )
        self.add_type_trait("feature_set", "kwiver:feature_set")
        self.add_type_trait("descriptor_set", "kwiver:descriptor_set")
        self.add_type_trait("track_set", "kwiver:track_set", VTC._convert_track_set_sptr )
        self.add_type_trait("homography_src_to_ref", "kwiver:s2r_homography")
        self.add_type_trait("homography_ref_to_src", "kwiver:r2s_homography")
        self.add_type_trait("image_file_name", "kwiver:image_file_name")
        self.add_type_trait("video_file_name", "kwiver:video_file_name")

        #          port-name   type-trait-name    description
        self.add_port_trait("timestamp", "timestamp", "Timestamp for input image")
        self.add_port_trait("image", "image", "Single frame input image")
        self.add_port_trait("mask", "mask", "Imput mask image")
        self.add_port_trait("feature_set", "feature_set", "Set of detected features")
        self.add_port_trait("descriptor_set", "descriptor_set", "Set of feature descriptors")
        self.add_port_trait("track_set", "track_set", "Set of feature tracks for stabilization")

        self.add_port_trait("homography_src_to_ref", "homography_src_to_ref", "Source image to ref image homography.")
        self.add_port_trait("image_file_name", "image_file_name", "Name of an image file. Usually a single frame of a video.")
        self.add_port_trait("video_file_name", "video_file_name", "Name of video file.")


    def add_type_trait(self, ttn, tn, conv = None):
        self._type_trait_set[ttn] = self.type_trait(ttn, tn, conv)

    def add_port_trait(self, nm, ttn, descr):
        # check to see if tn is in set below
        tt = self._type_trait_set.get(ttn)
        if tt == None:
            raise ValueError('type trait name "%" not registered' % (ttn))
        self._port_trait_set[nm] = self.port_trait(nm, tt, descr)

    def add_config_trait(self, name, key, default, descr):
        self._config_trait_set[name] = self.config_trait(name, key, default,descr)


    # ----------------------------------------------------------
    def declare_input_port_using_trait(self, ptn, flag):
        """
        Declare a port on the specified process.

        ptn: port trait name
        flag: required/optional flags

        There may be a better approach than passing the process as a parameter
        Maybe making this a class and deriving the user process from this.
        """
        port_trait = self._port_trait_set[ptn]
        if port_trait == None: raise ValueError('port trait name "%" not registered' % (ptn))

        self.declare_input_port(port_trait.name,
                                port_trait.type_trait.canonical_name,
                                flag,
                                port_trait.description)


    def declare_output_port_using_trait(self, ptn, flag):
        """
        Declare a port on the specified process.

        ptn: port trait name
        flag: required/optional flags

        There may be a better approach than passing the process as a parameter
        Maybe making this a class and deriving the user process from this.
        """
        port_trait = self._port_trait_set[ptn]
        if port_trait == None: raise ValueError('port trait name "%" not registered' % (ptn))

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

        This call is used to return managed types such as image_container, track_set.

        The raw datum contains the port data and other metadata.
        """
        pt = self._port_trait_set[ptn]
        if pt == None: raise ValueError('port trait name "%" not registered' % (ptn))

        pipeline_datum = self.grab_datum_from_port(pt.name)
        tt = pt.type_trait
        if tt.converter != None:
            data = tt.converter(pipeline_datum.get_datum_ptr())
            return data

        return pipeline_datum


    def grab_value_using_trait( self, ptn):
        """
        Get sprokit datum value from port. The caller is resonsible for
        converting the datum to an acceptable value.

        The value from the port datum is converted into a python type if
        there is a converter registered in sprokit. This is usually limited to
        fundimental types, such as int, double, bool, string, char
        """
        pt = self._port_trait_set[ptn]
        if pt == None: raise ValueError('port trait name "%" not registered' % (ptn))

        return self.grab_value_from_port(pt.name)


    def declare_config_using_trait(self, name):
        """
        Declare a process config entry from the named trait.
        """
        ct = self._config_trait_set[name]
        if ct == None: raise ValueError('config trait name "%" not registered' % (name))

        process.PythonProcess.declare_configuration_key(self, ct.key, ct.default, ct.description)


    def config_value_using_trait(self, name):
        """
        Get value from config using trait
        """
        ct = self._config_trait_set[name]
        if ct == None: raise ValueError('config trait name "%" not registered' % (name))

        return self.config_value(ct.name)
