"""
ckwg +29
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
"""

# make sure basic imports and algorithms required for burnour can be constructed
import kwiver
from kwiver.vital.algo import VideoInput, MetadataMapIO, BufferedMetadataFilter, ImageIO
from kwiver.vital import plugin_management
from kwiver.vital.config import empty_config


# instantiate any algorithm that burnout will try to create
def test_algorithms():
    vpm = plugin_management.plugin_manager_instance()
    vpm.load_all_plugins()

    assert VideoInput.create("ffmpeg") is not None

    config = empty_config()
    config["video_reader:type"] = "image_list"
    config["image_list"] = {"image_reader": {"type": "ocv"}}
    video_reader = VideoInput.set_nested_algo_configuration("video_reader", config)
    assert video_reader is not None

    # TODO how to test this ?
    # assert BufferedMetadataFilter.create("derive_metadata") is not None

    assert ImageIO.create("ocv") is not None

    for type_str in ("csv", "json", "klv-json"):
        config = empty_config()
        config["metadata_writer:type"] = type_str
        metadata_serializer = MetadataMapIO.set_nested_algo_configuration(
            "metadata_writer", config
        )
        assert metadata_serializer is not None
