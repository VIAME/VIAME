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
        import sprokit.pipeline_util.load
    except:
        test_error("Failed to import the load module")


def test_create(path_unused):
    from sprokit.pipeline_util import load

    load.Token()
    load.ConfigFlag()
    load.ConfigFlags()
    load.ConfigValue()
    load.ConfigValues()
    load.ConfigBlock()
    load.ProcessBlock()
    load.ConnectBlock()
    load.PipeBlock()
    load.PipeBlocks()
    load.ClusterConfig()
    load.ClusterInput()
    load.ClusterOutput()
    load.ClusterSubblock()
    load.ClusterSubblocks()
    load.ClusterBlock()
    load.ClusterDefineBlock()
    load.ClusterDefineBlocks()


def test_api_calls(path_unused):
    from sprokit.pipeline import config
    from sprokit.pipeline import process
    from sprokit.pipeline import process_factory
    from sprokit.pipeline_util import load

    o = load.ConfigValue()
    o.key
    o.value
    o.value = config.ConfigValue()

    o = load.ConfigBlock()
    o.key
    o.values
    o.values = load.ConfigValues()

    o = load.ProcessBlock()
    o.name
    o.type
    o.config_values
    o.name = process.ProcessName()
    o.type = process.ProcessType()
    o.config_values = load.ConfigValues()

    o = load.ConnectBlock()
    o.from_
    o.to
    o.from_ = process.PortAddr()
    o.to = process.PortAddr()

    o = load.PipeBlock()
    o.config = load.ConfigBlock()
    o.config
    o.process = load.ProcessBlock()
    o.process
    o.connect = load.ConnectBlock()
    o.connect

    o = load.ClusterConfig()
    o.description
    o.config_value
    o.description = config.ConfigDescription()
    o.config_value = load.ConfigValue()

    o = load.ClusterInput()
    o.description
    o.from_
    o.targets
    o.description = process.PortDescription()
    o.from_ = process.Port()
    o.targets = process.PortAddrs()

    o = load.ClusterOutput()
    o.description
    o.from_
    o.to
    o.description = process.PortDescription()
    o.from_ = process.PortAddr()
    o.to = process.Port()

    o = load.ClusterSubblock()
    o.config = load.ClusterConfig()
    if o.config is None:
        test_error("The 'config' is None when the cluster subblock is a config")
    if o.input is not None:
        test_error("The 'input' is not None when the cluster subblock is a config")
    if o.output is not None:
        test_error("The 'output' is not None when the cluster subblock is a config")
    o.input = load.ClusterInput()
    if o.config is not None:
        test_error("The 'config' is not None when the cluster subblock is an input")
    if o.input is None:
        test_error("The 'input' is None when the cluster subblock is an input")
    if o.output is not None:
        test_error("The 'output' is not None when the cluster subblock is an input")
    o.output = load.ClusterOutput()
    if o.config is not None:
        test_error("The 'config' is not None when the cluster subblock is an output")
    if o.input is not None:
        test_error("The 'input' is not None when the cluster subblock is an output")
    if o.output is None:
        test_error("The 'output' is None when the cluster subblock is an output")

    o = load.ClusterBlock()
    o.type
    o.description
    o.subblocks
    o.type = process.ProcessType()
    o.description = process_factory.ProcessDescription()
    o.subblocks = load.ClusterSubblocks()

    o = load.ClusterDefineBlock()
    o.config = load.ConfigBlock()
    if o.config is None:
        test_error("The 'config' is None when the pipe subblock is a config")
    if o.process is not None:
        test_error("The 'process' is not None when the pipe subblock is a config")
    if o.connect is not None:
        test_error("The 'connect' is not None when the pipe subblock is a config")
    if o.cluster is not None:
        test_error("The 'cluster' is not None when the pipe subblock is a config")
    o.process = load.ProcessBlock()
    if o.config is not None:
        test_error("The 'config' is not None when the pipe subblock is a process")
    if o.process is None:
        test_error("The 'process' is None when the pipe subblock is a process")
    if o.connect is not None:
        test_error("The 'connect' is not None when the pipe subblock is a process")
    if o.cluster is not None:
        test_error("The 'cluster' is not None when the pipe subblock is a process")
    o.connect = load.ConnectBlock()
    if o.config is not None:
        test_error("The 'config' is not None when the pipe subblock is a connection")
    if o.process is not None:
        test_error("The 'process' is not None when the pipe subblock is a connection")
    if o.connect is None:
        test_error("The 'connect' is None when the pipe subblock is a connection")
    if o.cluster is not None:
        test_error("The 'cluster' is not None when the pipe subblock is a connection")
    o.cluster = load.ClusterBlock()
    if o.config is not None:
        test_error("The 'config' is not None when the pipe subblock is a cluster")
    if o.process is not None:
        test_error("The 'process' is not None when the pipe subblock is a cluster")
    if o.connect is not None:
        test_error("The 'connect' is not None when the pipe subblock is a cluster")
    if o.cluster is None:
        test_error("The 'cluster' is None when the pipe subblock is a cluster")


def test_simple_pipeline(path):
    from sprokit.pipeline_util import load

    blocks = load.load_pipe_file(path)
    with open(path, 'r') as fin:
        load.load_pipe(fin)


def test_cluster_multiplier(path):
    from sprokit.pipeline_util import load

    blocks = load.load_cluster_file(path)
    with open(path, 'r') as fin:
        load.load_cluster(fin)


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
