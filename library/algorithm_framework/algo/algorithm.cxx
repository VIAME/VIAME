// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief base algorithm function implementations

// based on kwiver/vital/algo/algorithm.cxx
// @7e4920796821476afb9be3b31b23791a1e1e76b6

#include "algorithm.h"

namespace viame {

// ----------------------------------------------------------------------------
algorithm
::algorithm()
  : m_logger( viame::get_logger( "viame_algorithm_framework.algorithm" ) )
{}

// ----------------------------------------------------------------------------
void
algorithm
::attach_logger( std::string const& name )
{
  m_logger = viame::get_logger( name );
}

// ----------------------------------------------------------------------------
void
algorithm
::set_impl_name( const std::string& name )
{
  m_impl_name = name;
}

// ----------------------------------------------------------------------------
viame::logger_handle_t
algorithm
::logger() const
{
  return m_logger;
}

// ----------------------------------------------------------------------------
std::string
algorithm
::impl_name() const
{
  return m_impl_name;
}

// ----------------------------------------------------------------------------
config_block_sptr
algorithm
::get_configuration() const
{
  return viame::config_block::empty_config();
}

// ----------------------------------------------------------------------------
void
algorithm
::get_default_config( [[maybe_unused]] viame::config_block& cb )
{}

// ----------------------------------------------------------------------------
void
algorithm
::initialize()
{}

// ----------------------------------------------------------------------------
void
algorithm
::set_configuration_internal( [[maybe_unused]] config_block_sptr config )
{}

} // namespace viame
