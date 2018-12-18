# ckwg +28
# Copyright 2018 by Kitware, Inc.
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
"""
A small module to provide a consistent method for turning GPU list
argument strings to processes into a list of GPU indices and for
turning that into a Pytorch device.
"""

import torch

def gpu_list_desc(use_for=None):
    """
    Generate a description for a GPU list config trait.  The optional
    use_for argument, if passed, causes text to be included that says
    what task the GPU list will be used for.

    """
    return ('define which GPUs to use{}: "all", "None", or a comma-separated list, e.g. "1,2"'
            .format('' if use_for is None else ' for ' + use_for))

def parse_gpu_list(gpu_list_str):
    """
    Parse a string representing a list of GPU indices to a list of
    numeric GPU indices.  The indices should be separated by commas.
    Two special values are understood: the string "None" will produce
    an empty list, and the string "all" will produce the value None
    (which has a special meaning when picking a device).  Note that
    "None" is the only way to produce an empty list; an empty string
    won't work.
    """
    return ([] if gpu_list_str == 'None' else
            None if gpu_list_str == 'all' else
            list(map(int, gpu_list_str.split(','))))

def get_device(gpu_list=None):
    """Get a Pytorch device corresponding to one of the GPU indices listed
    in gpu_list.  If gpu_list is empty, get the device corresponding
    to the CPU instead.  If gpu_list is None (the default), enumerate
    the available GPU indices and pick one as though the list had been
    passed directly, except that in the case of there being no GPUs,
    an IndexError will be thrown.

    The return value is a pair of the device and a boolean that is
    true if the returned device is a GPU device.

    Note that we currently return the first listed device.

    """
    if gpu_list is None:
        gpu_list = list(range(torch.cuda.device_count()))
    elif not gpu_list:
        return torch.device('cpu'), False
    return torch.device('cuda:{}'.format(gpu_list[0])), True
