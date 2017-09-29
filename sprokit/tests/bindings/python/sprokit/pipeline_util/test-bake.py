#!/usr/bin/env python
#ckwg +28
# Copyright 2011-2013 by Kitware, Inc.
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


def test_import(path_unused):
    try:
        import sprokit.pipeline_util.bake
    except:
        test_error("Failed to import the bake module")


def test_simple_pipeline(path):
    from sprokit.pipeline import config
    from sprokit.pipeline import pipeline
    from sprokit.pipeline import modules
    from sprokit.pipeline_util import bake
    from sprokit.pipeline_util import load

    blocks = load.load_pipe_file(path)

    modules.load_known_modules()

    bake.bake_pipe_file(path)
    with open(path, 'r') as fin:
        bake.bake_pipe(fin)
    bake.bake_pipe_blocks(blocks)
    bake.extract_configuration(blocks)


def test_cluster_multiplier(path):
    from sprokit.pipeline import config
    from sprokit.pipeline import pipeline
    from sprokit.pipeline import modules
    from sprokit.pipeline_util import bake
    from sprokit.pipeline_util import load

    blocks = load.load_cluster_file(path)

    modules.load_known_modules()

    bake.bake_cluster_file(path)
    with open(path, 'r') as fin:
        bake.bake_cluster(fin)
    info = bake.bake_cluster_blocks(blocks)

    conf = config.empty_config()

    info.type()
    info.description()
    info.create()
    info.create(conf)

    bake.register_cluster(info)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 5:
        test_error("Expected four arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    pipeline_dir = sys.argv[4]

    path = os.path.join(pipeline_dir, '%s.pipe' % testname)

    from sprokit.test.test import *

    run_test(testname, find_tests(locals()), path)
