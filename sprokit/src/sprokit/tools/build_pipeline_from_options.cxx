/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include "build_pipeline_from_options.h"

#include "tool_io.h"
#include "tool_usage.h"

namespace sprokit {

// ------------------------------------------------------------------
build_pipeline_from_options::
build_pipeline_from_options()
{ }


// ------------------------------------------------------------------
build_pipeline_from_options::
build_pipeline_from_options( boost::program_options::variables_map const& vm,
                             boost::program_options::options_description const& desc )
{
  if (!vm.count("pipeline"))
  {
    std::cerr << "Error: pipeline option not set" << std::endl;
    tool_usage(EXIT_FAILURE, desc);
  }

  kwiver::vital::path_t const ipath = vm["pipeline"].as<kwiver::vital::path_t>();
  istream_t const istr = open_istream(ipath);

  /// \todo Include paths?

  this->load_pipeline(*istr, ipath);
  this->load_from_options(vm);
}


// ------------------------------------------------------------------
void
build_pipeline_from_options::
load_from_options( boost::program_options::variables_map const& vm )
{
  using namespace std::placeholders;  // for _1, _2, _3...

  // Load supplemental configuration files.
  if (vm.count("config"))
  {
    kwiver::vital::path_list_t const configs = vm["config"].as<kwiver::vital::path_list_t>();

    std::for_each(configs.begin(), configs.end(),
                  std::bind( &pipeline_builder::load_supplement, this, _1 ) );
  }

  // Insert lone setting variables from the command line.
  if (vm.count("setting"))
  {
    std::vector<std::string> const settings = vm["setting"].as<std::vector<std::string> >();

    std::for_each(settings.begin(), settings.end(),
                  std::bind( &pipeline_builder::add_setting, this, _1 ) );
  }
}

} // end namespace
