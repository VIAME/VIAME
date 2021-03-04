// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
  std::unique_ptr< cxxopts::Options > m_cmd_options;
  cxxopts::ParseResult* m_result;

  // Global options
  bool opt_detail {false};
  bool opt_help {false};
  bool opt_path_list {false};
  bool opt_brief {false};
  bool opt_modules {false};
  bool opt_files {false};
  bool opt_all {false};
  bool opt_algo {false};
  bool opt_process {false};
  bool opt_cluster {false};
  bool opt_scheduler {false};
  bool opt_summary {false};
  bool opt_attrs {false};
  bool opt_skip_relative {false};

  std::ostream* m_out_stream {nullptr};

  std::vector< std::string > opt_path;
  std::vector< std::string > opt_cluster_path;

  // Used to wrap large text blocks
  kwiver::vital::wrap_text_block m_wtb;

  // Fields used for filtering attributes
  bool opt_attr_filter {false};
  std::string opt_filter_attr;    // attribute name
  std::string opt_filter_regex;   // regex for attr value to match.

  // internal option for factory filtering
  bool opt_fact_filt {false};
  std::string opt_fact_regex;

  // internal option for instance type filtering
  bool opt_type_filt {false};
  std::string opt_type_regex;

  // Formatting type string. This is used as a suffix to the category
  // name to select specific explorer plugin when different formatting
  // styles are requested.
  std::string formatting_type;

  std::string opt_load_module;

  std::function<void(kwiver::vital::plugin_factory_handle_t const)> display_attr;

  priv() = default;

  virtual ~priv()
  { }
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
