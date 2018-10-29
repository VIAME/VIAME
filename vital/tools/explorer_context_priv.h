/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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

#ifndef VITAL_TOOLS_EXPLORER_CONTEXT_PRIV_H
#define VITAL_TOOLS_EXPLORER_CONTEXT_PRIV_H

#include <vital/util/wrap_text_block.h>
#include <functional>

namespace kwiver {
namespace vital {

class kwiver::vital::explorer_context::priv
{
public:

  // Collected command line args
  kwiversys::CommandLineArguments m_args;

  // Global options
  bool opt_detail;
  bool opt_help;
  bool opt_path_list;
  bool opt_brief;
  bool opt_modules;
  bool opt_files;
  bool opt_all;
  bool opt_algo;
  bool opt_process;
  bool opt_scheduler;
  bool opt_summary;
  bool opt_attrs;
  bool opt_skip_relative;

  std::ostream* m_out_stream;

  std::vector< std::string > opt_path;

  // Used to wrap large text blocks
  kwiver::vital::wrap_text_block m_wtb;

  // Fields used for filtering attributes
  bool opt_attr_filter;
  std::string opt_filter_attr;    // attribute name
  std::string opt_filter_regex;   // regex for attr value to match.

  // internal option for factory filtering
  bool opt_fact_filt;
  std::string opt_fact_regex;

  // internal option for instance type filtering
  bool opt_type_filt;
  std::string opt_type_regex;

  // Formatting type string. This is used as a suffix to the category
  // name to select specific explorer plugin when different formatting
  // styles are requested.
  std::string formatting_type;

  std::string opt_load_module;

  std::function<void(kwiver::vital::plugin_factory_handle_t const)> display_attr;

  priv()
  {
    opt_detail = false;
    opt_help = false;
    opt_path_list = false;
    opt_brief = false;
    opt_modules = false;
    opt_files = false;
    opt_all = false;
    opt_summary = false;
    opt_attrs = false;
    opt_skip_relative = false;

    opt_attr_filter = false;
    opt_fact_filt = false;

    m_out_stream = 0;
  }

  virtual ~priv()
  {
  }
};


// ==================================================================
class context_factory
  : public explorer_context
{
public:
  // -- CONSTRUCTORS --
  context_factory(kwiver::vital::explorer_context::priv* pp)
    : explorer_context( pp )
  { }

}; // end class context_factory


} } // end namespace

#endif // VITAL_TOOLS_EXPLORER_CONTEXT_PRIV_H
