/*ckwg +29
 * Copyright 2012-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sprokit/tools/tool_io.h>
#include <sprokit/tools/tool_main.h>
#include <sprokit/tools/tool_usage.h>
#include <sprokit/tools/build_pipeline_from_options.h>

#include <vital/config/config_block.h>

#include <sprokit/pipeline_util/export_pipe.h>

// Description of this program and why I would want to use it
static const std::string program_description(
"This tool reads a pipeline configuration file, applies the program options\n"
"and generates a \"compiled\" config file.\n"
"At its most basic, this tool will validate a pipeline\n"
"configuration, but it does so much more.  Specific pipeline\n"
"configurations can be generated from generic descriptions.\n"
"\n"
"Global config sections can ge inserted in the resulting configuration\n"
"file with the --setting option, with multiple options allowed on the\n"
"command line. For example, --setting master:value=FOO will generate a\n"
"config section:\n"
"\n"
"config master\n"
"  :value FOO\n"
"\n"
"The --config option specifies a file that contains additional\n"
"configuration parameters to be merged into the generated\n"
"configuration.\n"
"\n"
"Use the --include option to add additional directories to search for\n"
"included configuration files.\n"
"\n"
"The --pipeline option specifies the file that contains the main pipeline specification"
  );


// ------------------------------------------------------------------
int
sprokit_tool_main(int argc, char const* argv[])
{
  // Load all known modules
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  vpm.load_all_plugins();

  boost::program_options::options_description desc;
  desc
    .add(sprokit::tool_common_options())
    .add(sprokit::pipeline_common_options())
    .add(sprokit::pipeline_input_options())
    .add(sprokit::pipeline_output_options());

  boost::program_options::variables_map const vm = sprokit::tool_parse(argc, argv, desc,
    program_description );

  const sprokit::build_pipeline_from_options builder(vm, desc);

  sprokit::pipeline_t const pipe = builder.pipeline();
  kwiver::vital::config_block_sptr const config = builder.config();
  sprokit::pipe_blocks const blocks = builder.blocks();

  if (!pipe)
  {
    std::cerr << "Error: Unable to bake pipeline" << std::endl;

    return EXIT_FAILURE;
  }

  kwiver::vital::path_t const opath = vm["output"].as<kwiver::vital::path_t>();

  sprokit::ostream_t const ostr = sprokit::open_ostream(opath);

  sprokit::export_pipe exp( builder );
  exp.generate( *ostr.get() );

  return EXIT_SUCCESS;
}
