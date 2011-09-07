/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline_exception.h"

#include <sstream>
#include <string>

/**
 * \file pipeline_exception.cxx
 *
 * \brief Implementation of exceptions used within \link vistk::pipeline pipelines\endlink.
 */

namespace vistk
{

null_pipeline_config_exception
::null_pipeline_config_exception() throw()
  : pipeline_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a pipeline.";

  m_what = sstr.str();
}

null_pipeline_config_exception
::~null_pipeline_config_exception() throw()
{
}

null_process_addition_exception
::null_process_addition_exception() throw()
  : pipeline_addition_exception()
{
  std::ostringstream sstr;

  sstr << "A pipeline was given NULL as a process "
          "to add to the pipeline.";

  m_what = sstr.str();
}

null_process_addition_exception
::~null_process_addition_exception() throw()
{
}

duplicate_process_name_exception
::duplicate_process_name_exception(process::name_t const& name) throw()
  : pipeline_addition_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "A pipeline was given a process named "
          "\'" << m_name << "\' when one already "
          "exists by that name.";

  m_what = sstr.str();
}

duplicate_process_name_exception
::~duplicate_process_name_exception() throw()
{
}

no_such_process_exception
::no_such_process_exception(process::name_t const& name) throw()
  : pipeline_connection_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "A process named \'" << m_name << "\' "
          "was requested for a connection but "
          "it does not exist in the pipeline.";

  m_what = sstr.str();
}

no_such_process_exception
::~no_such_process_exception() throw()
{
}

connection_type_mismatch_exception
::connection_type_mismatch_exception(process::name_t const& upstream_name,
                                     process::port_t const& upstream_port,
                                     process::port_type_t const& upstream_type,
                                     process::name_t const& downstream_name,
                                     process::port_t const& downstream_port,
                                     process::port_type_t const& downstream_type) throw()
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
          "port on the \'" << m_upstream_name << "\' upstream "
          "and the \'" << m_downstream_port << "\' on the "
          "\'" << m_downstream_name << "\' connection mismatching "
          "types: up: \'" << m_upstream_type << "\' down: "
          "\'" << m_downstream_type << "\'.";

  m_what = sstr.str();
}

connection_type_mismatch_exception
::~connection_type_mismatch_exception() throw()
{
}

connection_flag_mismatch_exception
::connection_flag_mismatch_exception(process::name_t const& upstream_name,
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
          "port on the \'" << m_upstream_name << "\' upstream "
          "and the \'" << m_downstream_port << "\' on the "
          "\'" << m_downstream_name << "\' connection mismatching "
          "flags.";

  m_what = sstr.str();
}

connection_flag_mismatch_exception
::~connection_flag_mismatch_exception() throw()
{
}

no_processes_exception
::no_processes_exception() throw()
  : pipeline_setup_exception()
{
  std::ostringstream sstr;

  sstr << "The pipeline was setup without any processes in it.";

  m_what = sstr.str();
}

no_processes_exception
::~no_processes_exception() throw()
{
}

orphaned_processes_exception
::orphaned_processes_exception() throw()
  : pipeline_setup_exception()
{
  std::ostringstream sstr;

  sstr << "There are unconnected processes in the pipeline.";

  m_what = sstr.str();
}

orphaned_processes_exception
::~orphaned_processes_exception() throw()
{
}

no_such_group_exception
::no_such_group_exception(process::name_t const& name) throw()
  : pipeline_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "A group named \'" << m_name << "\' "
          "was requested it does not exist in "
          "the pipeline.";

  m_what = sstr.str();
}

no_such_group_exception
::~no_such_group_exception() throw()
{
}

no_such_group_port_exception
::no_such_group_port_exception(process::name_t const& name, process::port_t const& port) throw()
  : pipeline_exception()
  , m_name(name)
  , m_port(port)
{
  std::ostringstream sstr;

  sstr << "The \'" << m_port << "\' on the group "
          "named \'" << m_name << "\' was "
          "requested it does not exist.";

  m_what = sstr.str();
}

no_such_group_port_exception
::~no_such_group_port_exception() throw()
{
}

group_output_already_mapped_exception
::group_output_already_mapped_exception(process::name_t const& name,
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
          "\'" << m_port << "\' is already connected to "
          "the \'" << m_current_port << "\' port of the "
          "\'" << m_current_process << "\' process, but "
          "was attempted to be connected to the "
          "\'" << m_new_port << "\' port of the "
          "\'" << m_new_process << "\' process.";

  m_what = sstr.str();
}

group_output_already_mapped_exception
::~group_output_already_mapped_exception() throw()
{
}

}
