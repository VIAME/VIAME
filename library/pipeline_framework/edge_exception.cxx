// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "edge_exception.h"

#include <sstream>

/**
 * \file edge_exception.cxx
 *
 * \brief Implementation of exceptions used within \link sprokit::edge edges\endlink.
 */

namespace sprokit
{

edge_exception
::edge_exception() noexcept
  : pipeline_exception()
{
}

edge_exception
::~edge_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_edge_config_exception
::null_edge_config_exception() noexcept
  : edge_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a edge";

  m_what = sstr.str();
}

null_edge_config_exception
::~null_edge_config_exception() noexcept
{
}

// ----------------------------------------------------------------------------
datum_requested_after_complete
::datum_requested_after_complete() noexcept
  : edge_exception()
{
  std::ostringstream sstr;

  sstr << "A datum was requested after "
          "downstream indicated completion";

  m_what = sstr.str();
}

datum_requested_after_complete
::~datum_requested_after_complete() noexcept
{
}

// ----------------------------------------------------------------------------
edge_connection_exception
::edge_connection_exception() noexcept
  : edge_exception()
{
}

edge_connection_exception
::~edge_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
null_process_connection_exception
::null_process_connection_exception() noexcept
  : edge_connection_exception()
{
  std::ostringstream sstr;

  sstr << "An edge was given a NULL pointer to connect";

  m_what = sstr.str();
}

null_process_connection_exception
::~null_process_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
duplicate_edge_connection_exception
::duplicate_edge_connection_exception(process::name_t const& name, process::name_t const& new_name, std::string const& type) noexcept
  : edge_connection_exception()
  , m_name(name)
  , m_new_name(new_name)
{
  std::ostringstream sstr;

  sstr << "An edge was given a process for the "
       << type << " input (\'" << m_new_name
       << "\') when one already exists (\'" << m_name << "\')";

  m_what = sstr.str();
}

duplicate_edge_connection_exception
::~duplicate_edge_connection_exception() noexcept
{
}

// ----------------------------------------------------------------------------
input_already_connected_exception
::input_already_connected_exception(process::name_t const& name, process::name_t const& new_name) noexcept
  : duplicate_edge_connection_exception(name, new_name, "input")
{
}

input_already_connected_exception
::~input_already_connected_exception() noexcept
{
}

// ----------------------------------------------------------------------------
output_already_connected_exception
::output_already_connected_exception(process::name_t const& name, process::name_t const& new_name) noexcept
  : duplicate_edge_connection_exception(name, new_name, "output")
{
}

output_already_connected_exception
::~output_already_connected_exception() noexcept
{
}

}
