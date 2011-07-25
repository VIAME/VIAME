/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline_exception.h"

#include <sstream>

/**
 * \file pipeline_exception.cxx
 *
 * \brief Implementation of exceptions used within \link vistk::pipeline pipelines\endlink.
 */

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

connection_type_mismatch
::connection_type_mismatch(process::name_t const& upstream_name,
                           process::port_t const& upstream_port,
                           process::port_type_name_t const& upstream_type,
                           process::name_t const& downstream_name,
                           process::port_t const& downstream_port,
                           process::port_type_name_t const& downstream_type) throw()
  : pipeline_connection_exception()
  , m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_upstream_type(upstream_type)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
  , m_downstream_type(downstream_type)
{
  std::ostringstream sstr;

  sstr << "The connection between the \'" <<m_upstream_port << "\' "
       << "port on the \'" << m_upstream_name << "\' upstream "
       << "and the \'" << m_downstream_port << "\' on the "
       << "\'" << m_downstream_name << "\' connection mismatching "
       << "types: up: \'" << m_upstream_type << "\' down: "
       << "\'" << m_downstream_type << "\'.";

  m_what = sstr.str();
}

connection_type_mismatch
::~connection_type_mismatch() throw()
{
}

char const*
connection_type_mismatch
::what() const throw()
{
  return m_what.c_str();
}

connection_flag_mismatch
::connection_flag_mismatch(process::name_t const& upstream_name,
                           process::port_t const& upstream_port,
                           process::name_t const& downstream_name,
                           process::port_t const& downstream_port) throw()
  : pipeline_connection_exception()
  , m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
{
  std::ostringstream sstr;

  sstr << "The connection between the \'" <<m_upstream_port << "\' "
       << "port on the \'" << m_upstream_name << "\' upstream "
       << "and the \'" << m_downstream_port << "\' on the "
       << "\'" << m_downstream_name << "\' connection mismatching "
       << "flags.";

  m_what = sstr.str();
}

connection_flag_mismatch
::~connection_flag_mismatch() throw()
{
}

char const*
connection_flag_mismatch
::what() const throw()
{
  return m_what.c_str();
}

no_such_group
::no_such_group(process::name_t const& name) throw()
  : pipeline_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "A group named \'" << m_name << "\' "
       << "was requested it does not exist in "
       << "the pipeline.";

  m_what = sstr.str();
}

no_such_group
::~no_such_group() throw()
{
}

char const*
no_such_group
::what() const throw()
{
  return m_what.c_str();
}

no_such_group_port
::no_such_group_port(process::name_t const& name, process::port_t const& port) throw()
  : pipeline_exception()
  , m_name(name)
  , m_port(port)
{
  std::ostringstream sstr;

  sstr << "The \'" << m_port << "\' on the group "
       << "named \'" << m_name << "\' was "
       << "requested it does not exist.";

  m_what = sstr.str();
}

no_such_group_port
::~no_such_group_port() throw()
{
}

char const*
no_such_group_port
::what() const throw()
{
  return m_what.c_str();
}

group_output_already_mapped
::group_output_already_mapped(process::name_t const& name,
                              process::port_t const& port,
                              process::name_t const& current_process,
                              process::port_t const& current_port,
                              process::name_t const& new_process,
                              process::port_t const& new_port) throw()
  : pipeline_exception()
  , m_name(name)
  , m_port(port)
  , m_current_process(current_process)
  , m_current_port(current_port)
  , m_new_process(new_process)
  , m_new_port(new_port)
{
  std::ostringstream sstr;

  sstr << "The \'" << m_name << "\' group output port "
       << "\'" << m_port << "\' is already connected to "
       << "the \'" << m_current_port << "\' port of the "
       << "\'" << m_current_process << "\' process, but "
       << "was attempted to be connected to the "
       << "\'" << m_new_port << "\' port of the "
       << "\'" << m_new_process << "\' process.";

  m_what = sstr.str();
}

group_output_already_mapped
::~group_output_already_mapped() throw()
{
}

char const*
group_output_already_mapped
::what() const throw()
{
  return m_what.c_str();
}

} // end namespace vistk
