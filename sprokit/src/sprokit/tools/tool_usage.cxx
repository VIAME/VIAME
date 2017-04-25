/*ckwg +29
 * Copyright 2012-2013 by Kitware, Inc.
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

#include "tool_usage.h"
#include "tool_io.h"

#include <sprokit/version.h>

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/value_semantic.hpp>

#include <iostream>
#include <cstdlib>
#include <functional>

namespace sprokit
{

void
tool_usage(int ret, boost::program_options::options_description const& options)
{
  std::cerr << options << std::endl;

  exit(ret);
}


// ------------------------------------------------------------------
void
tool_version_message()
{
  std::cout << "sprokit " SPROKIT_VERSION_FULL << std::endl;
  std::cout << "Built with sprokit: " SPROKIT_VERSION << std::endl;
  std::cout << "Built from git:     "
#ifdef SPROKIT_BUILT_FROM_GIT
    "yes"
#else
    "no"
#endif
    << std::endl;
  std::cout << "Git hash:           " SPROKIT_GIT_HASH << std::endl;

  char const* const dirty = SPROKIT_GIT_DIRTY;
  bool const dirty_is_empty = (*dirty == '\0');
  char const* const is_dirty = (dirty_is_empty ? "no" : "yes");

  std::cout << "Dirty:              " << is_dirty << std::endl;
}


// ------------------------------------------------------------------
boost::program_options::options_description
tool_common_options()
{
  boost::program_options::options_description desc("General options");

  desc.add_options()
    ("help,h", "output help message and quit")
    ("version,V", "output version information")
  ;

  return desc;
}


// ------------------------------------------------------------------
boost::program_options::variables_map
tool_parse(int argc, char const* argv[], boost::program_options::options_description const& desc,
           std::string const& program_description)
{
  boost::program_options::variables_map vm;

  try
  {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  }
  catch (boost::program_options::unknown_option const& e)
  {
    std::cerr << "Error: unknown option " << e.get_option_name() << std::endl;

    tool_usage(EXIT_FAILURE, desc);
  }

  boost::program_options::notify(vm);

  if (vm.count("help"))
  {
    tool_version_message();
    std::cerr << program_description << std::endl;
    tool_usage(EXIT_SUCCESS, desc);
  }

  if (vm.count("version"))
  {
    tool_version_message();

    exit(EXIT_SUCCESS);
  }

  return vm;
}


// ------------------------------------------------------------------
boost::program_options::options_description
pipeline_common_options()
{
  boost::program_options::options_description desc("Common options");

  desc.add_options()
    ("config,c", boost::program_options::value<kwiver::vital::path_list_t>()->value_name("FILE"), "File containing supplemental configuration entries. ")
    ("setting,s", boost::program_options::value<std::vector<std::string> >()->value_name("VAR=VALUE"), "additional configuration")
    ("include,I", boost::program_options::value<kwiver::vital::path_list_t>()->value_name("DIR"),
     "a directory to be added to configuration include path")
  ;

  return desc;
}


// ------------------------------------------------------------------
boost::program_options::options_description
pipeline_input_options()
{
  boost::program_options::options_description desc("Input options");

  desc.add_options()
    ("pipeline,p", boost::program_options::value<kwiver::vital::path_t>()->value_name("FILE"),
     "Main input pipeline description file")
  ;

  return desc;
}


// ------------------------------------------------------------------
boost::program_options::options_description
pipeline_output_options()
{
  boost::program_options::options_description desc("Output options");

  desc.add_options()
    ("output,o", boost::program_options::value<kwiver::vital::path_t>()->value_name("FILE")->default_value("-"), "output path")
  ;

  return desc;
}


// ------------------------------------------------------------------
boost::program_options::options_description
pipeline_run_options()
{
  boost::program_options::options_description desc("Run options");

  desc.add_options()
    ("scheduler,S", boost::program_options::value<sprokit::scheduler::type_t>()->value_name("TYPE"), "scheduler type")
  ;

  return desc;
}


}
