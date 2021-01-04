// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pipe_display.h"

#include <vital/config/config_block.h>
#include <vital/util/string.h>
#include <vital/util/wrap_text_block.h>

#include <sprokit/pipeline_util/pipe_declaration_types.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/pipeline_exception.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>

#include <algorithm>
#include <iterator>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <cstdlib>

namespace sprokit {

namespace {

// ==================================================================
struct display_context
{
  display_context( std::ostream& str, bool print_loc )
    : m_ostr( str),
      m_opt_print_loc( print_loc )
  { }

  std::ostream& m_ostr;
  bool m_opt_print_loc;
};

// ==================================================================
class config_printer
{
public:
  config_printer( display_context& ctxt );
  ~config_printer() = default;

  void operator()( sprokit::config_pipe_block const& config_block );
  void operator()( sprokit::process_pipe_block const& process_block );
  void operator()( sprokit::connect_pipe_block const& connect_block ) const;

private:
  void print_config_value( sprokit::config_value_t const& config_value ) const;

  using process_set_t = std::set< sprokit::process::name_t >;

  display_context& m_ctxt;
};

config_printer
::config_printer( display_context& ctxt )
  : m_ctxt( ctxt )
{
}

// ==================================================================
class key_printer
{
public:
  key_printer( display_context& ctxt );
  ~key_printer() = default;

  void operator()( sprokit::config_value_t const& config_value ) const;

private:
  display_context& m_ctxt;
};

// ------------------------------------------------------------------
void
config_printer
::operator()( sprokit::config_pipe_block const& config_block )
{
  kwiver::vital::config_block_keys_t const& keys = config_block.key;
  sprokit::config_values_t const& values = config_block.values;

  kwiver::vital::config_block_key_t const key_path =
        kwiver::vital::join( keys,
                             kwiver::vital::config_block::block_sep() );

  // generate pipe level config block
  m_ctxt.m_ostr << "config " << key_path << std::endl;

  key_printer const printer( m_ctxt );

  std::for_each( values.begin(), values.end(), printer );
}

// ------------------------------------------------------------------
void
config_printer
::operator()( sprokit::process_pipe_block const& process_block )
{
  sprokit::process::name_t const& name = process_block.name;
  sprokit::process::type_t const& type = process_block.type;
  sprokit::config_values_t const& values = process_block.config_values;

  m_ctxt.m_ostr << "process " << name << std::endl
         << " :: " << type;

  if (m_ctxt.m_opt_print_loc)
  {
    m_ctxt.m_ostr << "  # from " << process_block.loc;
  }

  m_ctxt.m_ostr << std::endl;

  // Display process config items
  key_printer const printer( m_ctxt );
  std::for_each( values.begin(), values.end(), printer );
}

// ------------------------------------------------------------------
void
config_printer
::operator()( sprokit::connect_pipe_block const& connect_block ) const
{
  sprokit::process::port_addr_t const& upstream_addr = connect_block.from;
  sprokit::process::port_addr_t const& downstream_addr = connect_block.to;

  sprokit::process::name_t const& upstream_name = upstream_addr.first;
  sprokit::process::port_t const& upstream_port = upstream_addr.second;
  sprokit::process::name_t const& downstream_name = downstream_addr.first;
  sprokit::process::port_t const& downstream_port = downstream_addr.second;

  m_ctxt.m_ostr << "connect from " << upstream_name << "." << upstream_port << std::endl
         << "        to   " << downstream_name << "." << downstream_port << std::endl;

  if (m_ctxt.m_opt_print_loc)
  {
    m_ctxt.m_ostr << "  # from " << connect_block.loc;
  }
  m_ctxt.m_ostr << std::endl;
}

// ------------------------------------------------------------------
key_printer
::key_printer( display_context& ctxt )
  : m_ctxt( ctxt )
{
}

// ------------------------------------------------------------------
void
key_printer
::operator()( sprokit::config_value_t const& config_value ) const
{
  const auto& value = config_value.value;
  const auto& keys = config_value.key_path;
  const auto key_path = kwiver::vital::join( keys,
                 kwiver::vital::config_block::block_sep() );

  const auto& flags = config_value.flags;

  m_ctxt.m_ostr << "  " << key_path;

  if ( ! flags.empty() )
  {
    const auto flag_list = kwiver::vital::join( flags, "," );

    m_ctxt.m_ostr << "[" << flag_list << "]";
  }

  m_ctxt.m_ostr << " = " << value;

  if (m_ctxt.m_opt_print_loc)
  {
    m_ctxt.m_ostr << "  # from " << config_value.loc;
  }

  m_ctxt.m_ostr << std::endl;
}

} // end namespace

// ==================================================================
pipe_display
::pipe_display( std::ostream& ostr )
  : m_ostr( ostr )
{
}

pipe_display
::~pipe_display()
{ }

// ------------------------------------------------------------------
void
pipe_display
::display_pipe_blocks( const sprokit::pipe_blocks blocks )
{
  display_context local_ctxt( m_ostr, m_opt_print_loc );
  config_printer printer( local_ctxt );

  m_ostr << "Number of blocks in list: " << blocks.size() << std::endl;

  for ( auto b : blocks )
  {
    kwiver::vital::visit( printer, b );
  }
}

// ----------------------------------------------------------------------------
void
pipe_display
::print_loc( bool opt )
{
  m_opt_print_loc = opt;
}
} // end namespace
