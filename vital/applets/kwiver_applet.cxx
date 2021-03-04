// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "kwiver_applet.h"

#include "applet_context.h"

#include <vital/config/config_block_io.h>
#include <vital/util/get_paths.h>

namespace kwiver {
namespace tools {

namespace kv = ::kwiver::vital;

// ----------------------------------------------------------------------------
kwiver_applet::
kwiver_applet()
{
}

kwiver_applet::
~kwiver_applet()
{
}

// ----------------------------------------------------------------------------
kv::config_block_sptr
kwiver_applet::
find_configuration( std::string const& file_name )
{
  std::string prefix = kv::get_executable_path() + "/..";
  // empty application name and version means search only
  // KWIVER configuration paths
  return kv::read_config_file(file_name, "", "", prefix);
}

// ----------------------------------------------------------------------------
void
kwiver_applet::
add_command_options()
{
  // The default implementation assumes that the applet will do its
  // own arg parsing if it hasn't registered any specific args for the
  // tool_runner to parse.
  if ( m_context )
  {
    m_context->m_skip_command_args_parsing = true;
  }
  else
  {
    throw std::runtime_error( "Invalid context pointer" );
  }

}

// ----------------------------------------------------------------------------
cxxopts::ParseResult&
kwiver_applet::
command_args()
{
  if (m_context && m_context->m_result)
  {
    return *m_context->m_result;
  }

  throw std::runtime_error( "Invalid context pointer or command line results are not available." );
}

// ----------------------------------------------------------------------------
void
kwiver_applet::
initialize( kwiver::tools::applet_context* ctxt)
{
  m_context = ctxt;
  m_cmd_options.reset( new cxxopts::Options( applet_name(), "" ) );
}

// ----------------------------------------------------------------------------
std::string
kwiver_applet::
wrap_text( const std::string& text )
{
  if ( m_context )
  {
    return m_context->m_wtb.wrap_text( text );
  }

  throw std::runtime_error( "Invalid context pointer" );
}

// ----------------------------------------------------------------------------
const std::vector<std::string>&
kwiver_applet::
applet_args() const
{
  if ( m_context )
  {
    return m_context->m_argv;
  }

  throw std::runtime_error( "Invalid context pointer" );
}

// ----------------------------------------------------------------------------
const std::string&
kwiver_applet::
applet_name() const
{
  if ( m_context )
  {
    return m_context->m_applet_name;
  }

  throw std::runtime_error( "Invalid context pointer" );
}

} } // end namespace kwiver
