/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPROKIT_PIPELINE_PIPELINE_EXCEPTION_H
#define SPROKIT_PIPELINE_PIPELINE_EXCEPTION_H

#include "pipeline-config.h"
#include <vital/vital_config.h>

#include "process.h"
#include "types.h"

/**
 * \file pipeline_exception.h
 *
 * \brief Header for exceptions used within \link sprokit::pipeline pipelines\endlink.
 */

namespace sprokit
{

/**
 * \class pipeline_addition_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief The base exception thrown when adding processes to the pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pipeline_addition_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_addition_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    virtual ~pipeline_addition_exception() VITAL_NOTHROW;
};

/**
 * \class null_pipeline_config_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_pipeline_config_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_pipeline_config_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~null_pipeline_config_exception() VITAL_NOTHROW;
};

/**
 * \class add_after_setup_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a process or group is added to a pipeline after setup.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT add_after_setup_exception
  : public pipeline_addition_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     */
    add_after_setup_exception(process::name_t const& name) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~add_after_setup_exception() VITAL_NOTHROW;

    /// The name of the process.
    process::name_t const m_name;
};

/**
 * \class null_process_addition_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \c NULL is given as a \ref process to add to a \ref pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_process_addition_exception
  : public pipeline_addition_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_process_addition_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~null_process_addition_exception() VITAL_NOTHROW;
};

/**
 * \class duplicate_process_name_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref process with a duplicate name is added to the \ref pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT duplicate_process_name_exception
  : public pipeline_addition_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name requested.
     */
    duplicate_process_name_exception(process::name_t const& name) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~duplicate_process_name_exception() VITAL_NOTHROW;

    /// The name of the process.
    process::name_t const m_name;
};

/**
 * \class pipeline_removal_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief The base exception thrown when removing processes to the pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pipeline_removal_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_removal_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    virtual ~pipeline_removal_exception() VITAL_NOTHROW;
};

/**
 * \class remove_after_setup_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a process or group is removed from a pipeline after setup.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT remove_after_setup_exception
  : public pipeline_removal_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     */
    remove_after_setup_exception(process::name_t const& name) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~remove_after_setup_exception() VITAL_NOTHROW;

    /// The name of the process.
    process::name_t const m_name;
};

/**
 * \class reconfigure_before_setup_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline is reconfigured before setup.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT reconfigure_before_setup_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    reconfigure_before_setup_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    virtual ~reconfigure_before_setup_exception() VITAL_NOTHROW;
};

/**
 * \class pipeline_connection_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref pipeline due to connections.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pipeline_connection_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_connection_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    virtual ~pipeline_connection_exception() VITAL_NOTHROW;
};

/**
 * \class connection_after_setup_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection is requested after setup.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT connection_after_setup_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param upstream_name The name of the upstream process requested.
     * \param upstream_port The port on the upstream process requested.
     * \param downstream_name The name of the downstream process requested.
     * \param downstream_port The port on the downstream process requested.
     */
    connection_after_setup_exception(process::name_t const& upstream_name,
                                     process::port_t const& upstream_port,
                                     process::name_t const& downstream_name,
                                     process::port_t const& downstream_port) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~connection_after_setup_exception() VITAL_NOTHROW;

    /// The name of the upstream process requested.
    process::name_t const m_upstream_name;
    /// The name of the upstream port requested.
    process::port_t const m_upstream_port;
    /// The name of the downstream process requested.
    process::name_t const m_downstream_name;
    /// The name of the downstream port requested.
    process::port_t const m_downstream_port;
};

/**
 * \class disconnection_after_setup_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a disconnection is requested after setup.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT disconnection_after_setup_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param upstream_name The name of the upstream process requested.
     * \param upstream_port The port on the upstream process requested.
     * \param downstream_name The name of the downstream process requested.
     * \param downstream_port The port on the downstream process requested.
     */
    disconnection_after_setup_exception(process::name_t const& upstream_name,
                                        process::port_t const& upstream_port,
                                        process::name_t const& downstream_name,
                                        process::port_t const& downstream_port) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~disconnection_after_setup_exception() VITAL_NOTHROW;

    /// The name of the upstream process requested.
    process::name_t const m_upstream_name;
    /// The name of the upstream port requested.
    process::port_t const m_upstream_port;
    /// The name of the downstream process requested.
    process::name_t const m_downstream_name;
    /// The name of the downstream port requested.
    process::port_t const m_downstream_port;
};

/**
 * \class no_such_process_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref process is requested that does not exist in a \ref pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT no_such_process_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name requested.
     */
    no_such_process_exception(process::name_t const& name) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~no_such_process_exception() VITAL_NOTHROW;

    /// The name of the process requested.
    process::name_t const m_name;
};

/**
 * \class connection_dependent_type_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection sets a type on a port that is rejected.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT connection_dependent_type_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param upstream_name The name of the upstream process requested.
     * \param upstream_port The port on the upstream process requested.
     * \param downstream_name The name of the downstream process requested.
     * \param downstream_port The port on the downstream process requested.
     * \param type The type that was attempted to be set.
     * \param push_upstream True if upstream rejected the type.
     */
    connection_dependent_type_exception(process::name_t const& upstream_name,
                                        process::port_t const& upstream_port,
                                        process::name_t const& downstream_name,
                                        process::port_t const& downstream_port,
                                        process::port_type_t const& type,
                                        bool push_upstream) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~connection_dependent_type_exception() VITAL_NOTHROW;

    /// The name of the upstream process requested.
    process::name_t const m_upstream_name;
    /// The name of the upstream port requested.
    process::port_t const m_upstream_port;
    /// The name of the downstream process requested.
    process::name_t const m_downstream_name;
    /// The name of the downstream port requested.
    process::port_t const m_downstream_port;
    /// The name of the type that was attempted to be set.
    process::port_type_t const m_type;
    /// If set to \c true, upstream rejected the type, otherwise downstream did.
    bool const m_push_upstream;
};

/**
 * \class connection_dependent_type_cascade_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection sets a type on a port that is rejected due to propagation of the type.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT connection_dependent_type_cascade_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process which had a type set.
     * \param port The port on the process.
     * \param type The type that was attempted to be set.
     * \param upstream_name The upstream process.
     * \param upstream_port The port on upstream process that was attempted to be set to a type.
     * \param downstream_name The downstream process.
     * \param downstream_port The port on downstream process that was attempted to be set to a type.
     * \param cascade_type The type that was cascaded through the pipeline.
     * \param push_upstream True if upstream rejected the type.
     */
    connection_dependent_type_cascade_exception(process::name_t const& name,
                                                process::port_t const& port,
                                                process::port_type_t const& type,
                                                process::name_t const& upstream_name,
                                                process::port_t const& upstream_port,
                                                process::name_t const& downstream_name,
                                                process::port_t const& downstream_port,
                                                process::port_type_t const& cascade_type,
                                                bool push_upstream) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~connection_dependent_type_cascade_exception() VITAL_NOTHROW;

    /// The name of the process which had a type set.
    process::name_t const m_name;
    /// The name of the port which had a type set.
    process::port_t const m_port;
    /// The name of the type that was set.
    process::port_type_t const m_type;
    /// The name of the cascaded upstream process requested.
    process::name_t const m_upstream_name;
    /// The name of the cascaded upstream port requested.
    process::port_t const m_upstream_port;
    /// The name of the cascaded downstream process requested.
    process::name_t const m_downstream_name;
    /// The name of the cascaded downstream port requested.
    process::port_t const m_downstream_port;
    /// The name of the type that was rejected.
    process::port_type_t const m_cascade_type;
    /// If set to \c true, upstream rejected the type, otherwise downstream did.
    bool const m_push_upstream;
};

/**
 * \class connection_type_mismatch_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection is requested with type mismatched ports.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT connection_type_mismatch_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param upstream_name The name of the upstream process requested.
     * \param upstream_port The port on the upstream process requested.
     * \param upstream_type The type of the upstream process port.
     * \param downstream_name The name of the downstream process requested.
     * \param downstream_port The port on the downstream process requested.
     * \param downstream_type The type of the downstream process port.
     */
    connection_type_mismatch_exception(process::name_t const& upstream_name,
                                       process::port_t const& upstream_port,
                                       process::port_type_t const& upstream_type,
                                       process::name_t const& downstream_name,
                                       process::port_t const& downstream_port,
                                       process::port_type_t const& downstream_type) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~connection_type_mismatch_exception() VITAL_NOTHROW;

    /// The name of the upstream process requested.
    process::name_t const m_upstream_name;
    /// The name of the upstream port requested.
    process::port_t const m_upstream_port;
    /// The type available on the upstream port.
    process::port_type_t const m_upstream_type;
    /// The name of the downstream process requested.
    process::name_t const m_downstream_name;
    /// The name of the downstream port requested.
    process::port_t const m_downstream_port;
    /// The type available on the downstream port.
    process::port_type_t const m_downstream_type;
};

/**
 * \class connection_flag_mismatch_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection is requested with mismatched ports flags.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT connection_flag_mismatch_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param upstream_name The name of the upstream process requested.
     * \param upstream_port The port on the upstream process requested.
     * \param downstream_name The name of the upstream process requested.
     * \param downstream_port The port on the upstream process requested.
     */
    connection_flag_mismatch_exception(process::name_t const& upstream_name,
                                       process::port_t const& upstream_port,
                                       process::name_t const& downstream_name,
                                       process::port_t const& downstream_port) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~connection_flag_mismatch_exception() VITAL_NOTHROW;

    /// The name of the upstream process requested.
    process::name_t const m_upstream_name;
    /// The name of the upstream process requested.
    process::port_t const m_upstream_port;
    /// The name of the downstream process requested.
    process::name_t const m_downstream_name;
    /// The name of the downstream process requested.
    process::port_t const m_downstream_port;
};

/**
 * \class pipeline_setup_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref pipeline due to issues when setting up a pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pipeline_setup_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_setup_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    virtual ~pipeline_setup_exception() VITAL_NOTHROW;
};

/**
 * \class pipeline_duplicate_setup_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline is setup more than once.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pipeline_duplicate_setup_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_duplicate_setup_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~pipeline_duplicate_setup_exception() VITAL_NOTHROW;
};

/**
 * \class no_processes_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref pipeline has no processes in it.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT no_processes_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    no_processes_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~no_processes_exception() VITAL_NOTHROW;
};

/**
 * \class orphaned_processes_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref pipeline has orphaned processes in it.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT orphaned_processes_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    orphaned_processes_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~orphaned_processes_exception() VITAL_NOTHROW;
};

/**
 * \class not_a_dag_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref pipeline contains cycles.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT not_a_dag_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    not_a_dag_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~not_a_dag_exception() VITAL_NOTHROW;
};

/**
 * \class untyped_data_dependent_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref process contains a data-dependent port even after initialization.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT untyped_data_dependent_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the untyped port on the process.
     */
    untyped_data_dependent_exception(process::name_t const& name, process::port_t const& port) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~untyped_data_dependent_exception() VITAL_NOTHROW;

    /// The name of the process.
    process::name_t const m_name;
    /// The name of the untyped port.
    process::port_t const m_port;
};

/**
 * \class untyped_connection_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref process contains a data-dependent port even after initialization.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT untyped_connection_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    untyped_connection_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~untyped_connection_exception() VITAL_NOTHROW;
};

/**
 * \class frequency_mismatch_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when the frequencies within the pipeline cannot be resolved.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT frequency_mismatch_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param upstream_name The name of the upstream process requested.
     * \param upstream_port The port on the upstream process requested.
     * \param upstream_frequency The frequency of the upstream process.
     * \param upstream_port_frequency The frequency of the upstream port.
     * \param downstream_name The name of the downstream process requested.
     * \param downstream_port The port on the downstream process requested.
     * \param downstream_frequency The frequency of the downstream process.
     * \param downstream_port_frequency The frequency of the downstream port.
     */
    frequency_mismatch_exception(process::name_t const& upstream_name,
                                 process::port_t const& upstream_port,
                                 process::port_frequency_t const& upstream_frequency,
                                 process::port_frequency_t const& upstream_port_frequency,
                                 process::name_t const& downstream_name,
                                 process::port_t const& downstream_port,
                                 process::port_frequency_t const& downstream_frequency,
                                 process::port_frequency_t const& downstream_port_frequency) VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~frequency_mismatch_exception() VITAL_NOTHROW;

    /// The name of the upstream process requested.
    process::name_t const m_upstream_name;
    /// The port on the upstream process requested.
    process::port_t const m_upstream_port;
    /// The frequency of the upstream process.
    process::port_frequency_t const m_upstream_frequency;
    /// The frequency of the upstream port.
    process::port_frequency_t const m_upstream_port_frequency;
    /// The name of the downstream process requested.
    process::name_t const m_downstream_name;
    /// The port on the downstream process requested.
    process::port_t const m_downstream_port;
    /// The frequency of the downstream process.
    process::port_frequency_t const m_downstream_frequency;
    /// The frequency of the downstream port.
    process::port_frequency_t const m_downstream_port_frequency;
};

/**
 * \class reset_running_pipeline_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline is reset while it is running.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT reset_running_pipeline_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    reset_running_pipeline_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~reset_running_pipeline_exception() VITAL_NOTHROW;
};

/**
 * \class pipeline_not_setup_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline that has not been setup is started.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pipeline_not_setup_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_not_setup_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~pipeline_not_setup_exception() VITAL_NOTHROW;
};

/**
 * \class pipeline_not_ready_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline that did not succeed in setting up is started.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT pipeline_not_ready_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_not_ready_exception() VITAL_NOTHROW;
    /**
     * \brief Destructor.
     */
    ~pipeline_not_ready_exception() VITAL_NOTHROW;
};

}

#endif // SPROKIT_PIPELINE_PIPELINE_EXCEPTION_H
