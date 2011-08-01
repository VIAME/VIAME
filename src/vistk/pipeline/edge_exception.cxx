/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "edge_exception.h"

#include <sstream>

/**
 * \file edge_exception.cxx
 *
 * \brief Implementation of exceptions used within \link vistk::edge edges\endlink.
 */

namespace vistk
{

null_process_connection_exception
::null_process_connection_exception() throw()
  : edge_connection_exception()
{
  std::ostringstream sstr;

  sstr << "An edge was given a NULL pointer to connect.";

  m_what = sstr.str();
}

null_process_connection_exception
::~null_process_connection_exception() throw()
{
}

char const*
null_process_connection_exception
::what() const throw()
{
  return m_what.c_str();
}

duplicate_edge_connection_exception
::duplicate_edge_connection_exception(process::name_t const& process, process::name_t const& new_process, std::string const& type) throw()
  : edge_connection_exception()
  , m_process(process)
  , m_new_process(new_process)
{
  std::ostringstream sstr;

  sstr << "An edge was given a process for the "
       << type << " input "
       << "(\'" << m_new_process << "\') when one already "
       << "exists (\'" << m_process << "\').";

  m_what = sstr.str();
}

duplicate_edge_connection_exception
::~duplicate_edge_connection_exception() throw()
{
}

char const*
duplicate_edge_connection_exception
::what() const throw()
{
  return m_what.c_str();
}

input_already_connected_exception
::input_already_connected_exception(process::name_t const& process, process::name_t const& new_process) throw()
  : duplicate_edge_connection_exception(process, new_process, "input")
{
}

input_already_connected_exception
::~input_already_connected_exception() throw()
{
}

output_already_connected_exception
::output_already_connected_exception(process::name_t const& process, process::name_t const& new_process) throw()
  : duplicate_edge_connection_exception(process, new_process, "output")
{
}

output_already_connected_exception
::~output_already_connected_exception() throw()
{
}

} // end namespace vistk
