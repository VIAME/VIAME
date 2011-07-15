/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline_exception.h"

#include <sstream>

namespace vistk
{

null_process_addition
::null_process_addition() throw()
  : pipeline_addition_exception()
{
  std::ostringstream sstr;

  sstr << "A pipeline was given NULL as a process "
       << "to add to the pipeline.";

  m_what = sstr.str();
}

null_process_addition
::~null_process_addition() throw()
{
}

char const*
null_process_addition
::what() const throw()
{
  return m_what.c_str();
}

duplicate_process_name
::duplicate_process_name(process::name_t const& name) throw()
  : pipeline_addition_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "A pipeline was given a process named "
       << "\'" << m_name << "\' when one already "
       << "exists by that name.";

  m_what = sstr.str();
}

duplicate_process_name
::~duplicate_process_name() throw()
{
}

char const*
duplicate_process_name
::what() const throw()
{
  return m_what.c_str();
}

null_edge_connection
::null_edge_connection(process::name_t const& upstream_process,
                       process::port_t const& upstream_port,
                       process::name_t const& downstream_process,
                       process::port_t const& downstream_port) throw()
  : pipeline_connection_exception()
  , m_upstream_process(upstream_process)
  , m_upstream_port(upstream_port)
  , m_downstream_process(downstream_process)
  , m_downstream_port(downstream_port)
{
  std::ostringstream sstr;

  sstr << "The edge connecting the upstream port "
       << "\'" << m_upstream_process << "."
       << m_upstream_port << "\' to "
       << "\'" << m_downstream_process << "."
       << m_downstream_port << "\' was NULL.";

  m_what = sstr.str();
}

null_edge_connection
::~null_edge_connection() throw()
{
}

char const*
null_edge_connection
::what() const throw()
{
  return m_what.c_str();
}

no_such_process
::no_such_process(process::name_t const& name) throw()
  : pipeline_connection_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "A process named \'" << m_name << "\' "
       << "was requested for a connection but "
       << "it does not exist in the pipeline.";

  m_what = sstr.str();
}

no_such_process
::~no_such_process() throw()
{
}

char const*
no_such_process
::what() const throw()
{
  return m_what.c_str();
}

} // end namespace vistk
