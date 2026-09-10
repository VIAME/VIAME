// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pipeline_exception.h"

#include <sstream>
#include <string>

/**
 * \file pipeline_exception.cxx
 *
 * \brief Implementation of exceptions used within \link sprokit::pipeline pipelines\endlink.
 */

namespace sprokit
{

pipeline_addition_exception
::pipeline_addition_exception() noexcept
  : pipeline_exception()
{
}

pipeline_addition_exception
::~pipeline_addition_exception() noexcept
{
}

null_pipeline_config_exception
::null_pipeline_config_exception() noexcept
  : pipeline_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a pipeline";

  m_what = sstr.str();
}

null_pipeline_config_exception
::~null_pipeline_config_exception() noexcept
{
}

add_after_setup_exception
::add_after_setup_exception(process::name_t const& name) noexcept
  : pipeline_addition_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process named \'" << m_name << "\' "
          "was added to the pipeline after it was setup";

  m_what = sstr.str();
}

add_after_setup_exception
::~add_after_setup_exception() noexcept
{
}

null_process_addition_exception
::null_process_addition_exception() noexcept
  : pipeline_addition_exception()
{
  std::ostringstream sstr;

  sstr << "A pipeline was given NULL as a process "
          "to add to the pipeline";

  m_what = sstr.str();
}

null_process_addition_exception
::~null_process_addition_exception() noexcept
{
}

pipeline_setup_exception
::pipeline_setup_exception() noexcept
  : pipeline_exception()
{
}

pipeline_setup_exception
::~pipeline_setup_exception() noexcept
{
}

duplicate_process_name_exception
::duplicate_process_name_exception(process::name_t const& name) noexcept
  : pipeline_addition_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "A pipeline was given a process named "
          "\'" << m_name << "\' when one already "
          "exists by that name";

  m_what = sstr.str();
}

duplicate_process_name_exception
::~duplicate_process_name_exception() noexcept
{
}

pipeline_removal_exception
::pipeline_removal_exception() noexcept
  : pipeline_exception()
{
}

pipeline_removal_exception
::~pipeline_removal_exception() noexcept
{
}

remove_after_setup_exception
::remove_after_setup_exception(process::name_t const& name) noexcept
  : pipeline_removal_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "The process named \'" << m_name << "\' "
          "was removed from the pipeline after it was setup";

  m_what = sstr.str();
}

remove_after_setup_exception
::~remove_after_setup_exception() noexcept
{
}

reconfigure_before_setup_exception
::reconfigure_before_setup_exception() noexcept
  : pipeline_exception()
{
  std::ostringstream sstr;

  sstr << "A pipeline was reconfigured before it was setup";

  m_what = sstr.str();
}

reconfigure_before_setup_exception
::~reconfigure_before_setup_exception() noexcept
{
}

pipeline_connection_exception
::pipeline_connection_exception() noexcept
  : pipeline_exception()
{
}

pipeline_connection_exception
::~pipeline_connection_exception() noexcept
{
}

connection_after_setup_exception
::connection_after_setup_exception(process::name_t const& upstream_name,
                                   process::port_t const& upstream_port,
                                   process::name_t const& downstream_name,
                                   process::port_t const& downstream_port) noexcept
  : pipeline_connection_exception()
  , m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
{
  std::ostringstream sstr;

  sstr << "The connection from "
          "\'" << m_upstream_name << "." << m_upstream_port << "\' to "
          "\'" << m_downstream_name << "." << m_downstream_port << "\', "
          "was requested after the pipeline was setup";

  m_what = sstr.str();
}

connection_after_setup_exception
::~connection_after_setup_exception() noexcept
{
}

disconnection_after_setup_exception
::disconnection_after_setup_exception(process::name_t const& upstream_name,
                                      process::port_t const& upstream_port,
                                      process::name_t const& downstream_name,
                                      process::port_t const& downstream_port) noexcept
  : pipeline_connection_exception()
  , m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
{
  std::ostringstream sstr;

  sstr << "The connection from "
          "\'" << m_upstream_name << "." << m_upstream_port << "\' to "
          "\'" << m_downstream_name << "." << m_downstream_port << "\', "
          "was requested to be disconnected after the pipeline was setup";

  m_what = sstr.str();
}

disconnection_after_setup_exception
::~disconnection_after_setup_exception() noexcept
{
}

no_such_process_exception
::no_such_process_exception(process::name_t const& name) noexcept
  : pipeline_connection_exception()
  , m_name(name)
{
  std::ostringstream sstr;

  sstr << "A process named \'" << m_name << "\' "
          "was requested for a connection but "
          "it does not exist in the pipeline";

  m_what = sstr.str();
}

no_such_process_exception
::~no_such_process_exception() noexcept
{
}

connection_dependent_type_exception
::connection_dependent_type_exception(process::name_t const& upstream_name,
                                      process::port_t const& upstream_port,
                                      process::name_t const& downstream_name,
                                      process::port_t const& downstream_port,
                                      process::port_type_t const& type,
                                      bool push_upstream) noexcept
  : pipeline_connection_exception()
  , m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
  , m_type(type)
  , m_push_upstream(push_upstream)
{
  std::ostringstream sstr;

  process::name_t const& error_name = (m_push_upstream ? m_upstream_name : m_downstream_name);

  sstr << "When connecting "
          "\'" << m_upstream_name << "." << m_upstream_port << "\' to "
          "\'" << m_downstream_name << "." << m_downstream_port << "\', "
          "the process \'" << error_name << "\' rejected the type "
          "\'" << m_type << "\'";

  m_what = sstr.str();
}

connection_dependent_type_exception
::~connection_dependent_type_exception() noexcept
{
}

connection_dependent_type_cascade_exception
::connection_dependent_type_cascade_exception(process::name_t const& name,
                                              process::port_t const& port,
                                              process::port_type_t const& type,
                                              process::name_t const& upstream_name,
                                              process::port_t const& upstream_port,
                                              process::name_t const& downstream_name,
                                              process::port_t const& downstream_port,
                                              process::port_type_t const& cascade_type,
                                              bool push_upstream) noexcept
  : pipeline_connection_exception()
  , m_name(name)
  , m_port(port)
  , m_type(type)
  , m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
  , m_cascade_type(cascade_type)
  , m_push_upstream(push_upstream)
{
  std::ostringstream sstr;

  process::name_t const& error_name = (m_push_upstream ? m_upstream_name : m_downstream_name);

  sstr << "When setting the type of the port "
          "\'" << m_name << "." << m_port << "\' to "
          "\'" << m_type << "\', the setting of the connection from "
          "\'" << m_upstream_name << "." << m_upstream_port << "\' to "
          "\'" << m_downstream_name << "." << m_downstream_port << "\' "
          "was set to the type \'" << m_cascade_type << "\' which was "
          "rejected by the \'" << error_name << "\' process";

  m_what = sstr.str();
}

connection_dependent_type_cascade_exception
::~connection_dependent_type_cascade_exception() noexcept
{
}

connection_type_mismatch_exception
::connection_type_mismatch_exception(process::name_t const& upstream_name,
                                     process::port_t const& upstream_port,
                                     process::port_type_t const& upstream_type,
                                     process::name_t const& downstream_name,
                                     process::port_t const& downstream_port,
                                     process::port_type_t const& downstream_type) noexcept
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
          "\'" << m_downstream_type << "\'";

  m_what = sstr.str();
}

connection_type_mismatch_exception
::~connection_type_mismatch_exception() noexcept
{
}

connection_flag_mismatch_exception
::connection_flag_mismatch_exception(process::name_t const& upstream_name,
                                     process::port_t const& upstream_port,
                                     process::name_t const& downstream_name,
                                     process::port_t const& downstream_port) noexcept
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
          "\'" << m_downstream_name << "\' connection have "
          "mismatching flags";

  m_what = sstr.str();
}

connection_flag_mismatch_exception
::~connection_flag_mismatch_exception() noexcept
{
}

pipeline_duplicate_setup_exception
::pipeline_duplicate_setup_exception() noexcept
  : pipeline_setup_exception()
{
  std::ostringstream sstr;

  sstr << "The pipeline was setup a second time";

  m_what = sstr.str();
}

pipeline_duplicate_setup_exception
::~pipeline_duplicate_setup_exception() noexcept
{
}

no_processes_exception
::no_processes_exception() noexcept
  : pipeline_setup_exception()
{
  std::ostringstream sstr;

  sstr << "The pipeline was setup without any processes in it";

  m_what = sstr.str();
}

no_processes_exception
::~no_processes_exception() noexcept
{
}

orphaned_processes_exception
::orphaned_processes_exception() noexcept
  : pipeline_setup_exception()
{
  std::ostringstream sstr;

  sstr << "There are unconnected processes in the pipeline";

  m_what = sstr.str();
}

orphaned_processes_exception
::~orphaned_processes_exception() noexcept
{
}

not_a_dag_exception
::not_a_dag_exception() noexcept
  : pipeline_setup_exception()
{
  std::ostringstream sstr;

  sstr << "The pipeline contains a cycle in it. Backwards "
          "edges should only be connected to input ports "
          "which have the process::flag_input_nodep flag on them";

  m_what = sstr.str();
}

not_a_dag_exception
::~not_a_dag_exception() noexcept
{
}

untyped_data_dependent_exception
::untyped_data_dependent_exception(process::name_t const& name, process::port_t const& port) noexcept
  : pipeline_setup_exception()
  , m_name(name)
  , m_port(port)
{
  std::ostringstream sstr;

  sstr << "After configure, the \'" << m_port << "\' "
          "port on the \'" << m_name << "\' process "
          "was still marked as data-dependent";

  m_what = sstr.str();
}

untyped_data_dependent_exception
::~untyped_data_dependent_exception() noexcept
{
}

untyped_connection_exception
::untyped_connection_exception() noexcept
  : pipeline_setup_exception()
{
  std::ostringstream sstr;

  sstr << "After all of the processes have been initialized, "
          "there are still untyped connections in the pipeline";

  m_what = sstr.str();
}

untyped_connection_exception
::~untyped_connection_exception() noexcept
{
}

frequency_mismatch_exception
::frequency_mismatch_exception(process::name_t const& upstream_name,
                               process::port_t const& upstream_port,
                               process::port_frequency_t const& upstream_frequency,
                               process::port_frequency_t const& upstream_port_frequency,
                               process::name_t const& downstream_name,
                               process::port_t const& downstream_port,
                               process::port_frequency_t const& downstream_frequency,
                               process::port_frequency_t const& downstream_port_frequency) noexcept
  : m_upstream_name(upstream_name)
  , m_upstream_port(upstream_port)
  , m_upstream_frequency(upstream_frequency)
  , m_upstream_port_frequency(upstream_port_frequency)
  , m_downstream_name(downstream_name)
  , m_downstream_port(downstream_port)
  , m_downstream_frequency(downstream_frequency)
  , m_downstream_port_frequency(downstream_port_frequency)
{
  std::ostringstream sstr;

  process::port_frequency_t const up_freq = m_upstream_frequency * m_upstream_port_frequency;
  process::port_frequency_t const down_freq = m_downstream_frequency * m_downstream_port_frequency;

  sstr << "The connection from "
          "\'" << m_upstream_name << "." << m_upstream_port << "\' to "
          "\'" << m_downstream_name << "." << m_downstream_port << "\', "
          "has a frequency mismatch where upstream is at "
       << up_freq << " and downstream is at " << down_freq;

  m_what = sstr.str();
}

frequency_mismatch_exception
::~frequency_mismatch_exception() noexcept
{
}

reset_running_pipeline_exception
::reset_running_pipeline_exception() noexcept
{
  std::ostringstream sstr;

  sstr << "A pipeline was running when a reset was attempted";

  m_what = sstr.str();
}

reset_running_pipeline_exception
::~reset_running_pipeline_exception() noexcept
{
}

pipeline_not_setup_exception
::pipeline_not_setup_exception() noexcept
  : pipeline_exception()
{
  std::ostringstream sstr;

  sstr << "The pipeline has not been setup";

  m_what = sstr.str();
}

pipeline_not_setup_exception
::~pipeline_not_setup_exception() noexcept
{
}

pipeline_not_ready_exception
::pipeline_not_ready_exception() noexcept
  : pipeline_exception()
{
  std::ostringstream sstr;

  sstr << "The pipeline has not been successfully setup";

  m_what = sstr.str();
}

pipeline_not_ready_exception
::~pipeline_not_ready_exception() noexcept
{
}

}
