/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PIPELINE_EXCEPTION_H
#define VISTK_PIPELINE_PIPELINE_EXCEPTION_H

#include "pipeline-config.h"

#include "process.h"
#include "types.h"

/**
 * \file pipeline_exception.h
 *
 * \brief Header for exceptions used within \link vistk::pipeline pipelines\endlink.
 */

namespace vistk
{

/**
 * \class pipeline_addition_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief The base exception thrown when adding processes to the pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_addition_exception
  : public pipeline_exception
{
};

/**
 * \class null_pipeline_config_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when \c NULL \ref config is passed to a pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_pipeline_config_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_pipeline_config_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_pipeline_config_exception() throw();
};

/**
 * \class add_process_after_setup_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a process or group is added to a pipeline after setup.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT add_after_setup_exception
  : public pipeline_addition_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param is_process True if the addition is a process, false if it was a group.
     */
    add_after_setup_exception(process::name_t const& name, bool is_process) throw();
    /**
     * \brief Destructor.
     */
    ~add_after_setup_exception() throw();

    /// The name of the process.
    process::name_t const m_name;
    /// True if the addition is a process, false if it was a group.
    bool const m_is_process;
};

/**
 * \class null_process_addition_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \c NULL is given as a \ref process to add to a \ref pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_process_addition_exception
  : public pipeline_addition_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_process_addition_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_process_addition_exception() throw();
};

/**
 * \class duplicate_process_name_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref process with a duplicate name is added to the \ref pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT duplicate_process_name_exception
  : public pipeline_addition_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name requested.
     */
    duplicate_process_name_exception(process::name_t const& name) throw();
    /**
     * \brief Destructor.
     */
    ~duplicate_process_name_exception() throw();

    /// The name of the process.
    process::name_t const m_name;
};

/**
 * \class pipeline_connection_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref pipeline due to connections.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_connection_exception
  : public pipeline_exception
{
};

/**
 * \class connection_after_setup_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection is requested after setup.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT connection_after_setup_exception
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
                                     process::port_t const& downstream_port) throw();
    /**
     * \brief Destructor.
     */
    ~connection_after_setup_exception() throw();

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
 * \class no_such_process_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref process is requested that does not exist in a \ref pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_process_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name requested.
     */
    no_such_process_exception(process::name_t const& name) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_process_exception() throw();

    /// The name of the process requested.
    process::name_t const m_name;
};

/**
 * \class connection_dependent_type_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection sets a type on a port that is rejected.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT connection_dependent_type_exception
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
                                        bool push_upstream) throw();
    /**
     * \brief Destructor.
     */
    ~connection_dependent_type_exception() throw();

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
 * \class connection_dependent_type_cascade_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection sets a type on a port that is rejected due to propogation of the type.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT connection_dependent_type_cascade_exception
  : public pipeline_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process which had a type set.
     * \param port The port on the process.
     * \param type The type that was attempted to be set.
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
                                                bool push_upstream) throw();
    /**
     * \brief Destructor.
     */
    ~connection_dependent_type_cascade_exception() throw();

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
 * \class connection_type_mismatch_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection is requested with type mismatched ports.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT connection_type_mismatch_exception
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
                                       process::port_type_t const& downstream_type) throw();
    /**
     * \brief Destructor.
     */
    ~connection_type_mismatch_exception() throw();

    /// The name of the upstream process requested.
    process::name_t const m_upstream_name;
    /// The name of the upstream port requested.
    process::port_t const m_upstream_port;
    /// The name of the type available on the upstream port.
    process::port_type_t const m_upstream_type;
    /// The name of the downstream process requested.
    process::name_t const m_downstream_name;
    /// The name of the downstream port requested.
    process::port_t const m_downstream_port;
    /// The name of the type available on the downstream port.
    process::port_type_t const m_downstream_type;
};

/**
 * \class connection_flag_mismatch_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a connection is requested with mismatched ports flags.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT connection_flag_mismatch_exception
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
                                       process::port_t const& downstream_port) throw();
    /**
     * \brief Destructor.
     */
    ~connection_flag_mismatch_exception() throw();

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
 * \class pipeline_setup_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref pipeline due to issues when setting up a pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_setup_exception
  : public pipeline_exception
{
};

/**
 * \class pipeline_duplicate_setup_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline is setup more than once.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_duplicate_setup_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_duplicate_setup_exception() throw();
    /**
     * \brief Destructor.
     */
    ~pipeline_duplicate_setup_exception() throw();
};

/**
 * \class no_processes_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref pipeline has no processes in it.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_processes_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    no_processes_exception() throw();
    /**
     * \brief Destructor.
     */
    ~no_processes_exception() throw();
};

/**
 * \class orphaned_processes_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref pipeline has orphaned processes in it.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT orphaned_processes_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    orphaned_processes_exception() throw();
    /**
     * \brief Destructor.
     */
    ~orphaned_processes_exception() throw();
};

/**
 * \class not_a_dag_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref pipeline contains cycles.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT not_a_dag_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    not_a_dag_exception() throw();
    /**
     * \brief Destructor.
     */
    ~not_a_dag_exception() throw();
};

/**
 * \class untyped_data_dependent_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref process contains a data-dependent port even after initialization.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT untyped_data_dependent_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the untyped port on the process.
     */
    untyped_data_dependent_exception(process::name_t const& name, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    ~untyped_data_dependent_exception() throw();

    /// The name of the process.
    process::name_t const m_name;
    /// The name of the untyped port.
    process::port_t const m_port;
};

/**
 * \class untyped_connection_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a \ref process contains a data-dependent port even after initialization.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT untyped_connection_exception
  : public pipeline_setup_exception
{
  public:
    /**
     * \brief Constructor.
     */
    untyped_connection_exception() throw();
    /**
     * \brief Destructor.
     */
    ~untyped_connection_exception() throw();
};

/**
 * \class no_such_group_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a group is requested that does not exist in a \ref pipeline.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_group_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name requested.
     */
    no_such_group_exception(process::name_t const& name) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_group_exception() throw();

    /// The name of the group requested.
    process::name_t const m_name;
};

/**
 * \class no_such_group_port_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a port on a group is requested that does not exist.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_group_port_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name requested.
     * \param port The port requested.
     */
    no_such_group_port_exception(process::name_t const& name, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_group_port_exception() throw();

    /// The name of the group requested.
    process::name_t const m_name;
    /// The name of the port requested.
    process::port_t const m_port;
};

/**
 * \class group_output_already_mapped_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when an output port on a group is attempted to be remapped.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT group_output_already_mapped_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name requested.
     * \param port The port requested.
     * \param current_process The current process mapped.
     * \param current_port The current port mapped.
     * \param new_process The process requested to be mapped.
     * \param new_port The port requested to be mapped.
     */
    group_output_already_mapped_exception(process::name_t const& name,
                                          process::port_t const& port,
                                          process::name_t const& current_process,
                                          process::port_t const& current_port,
                                          process::name_t const& new_process,
                                          process::port_t const& new_port) throw();
    /**
     * \brief Destructor.
     */
    ~group_output_already_mapped_exception() throw();

    /// The name of the group requested.
    process::name_t const m_name;
    /// The name of the port requested.
    process::port_t const m_port;
    /// The name of the current process requested.
    process::name_t const m_current_process;
    /// The name of the current port requested.
    process::port_t const m_current_port;
    /// The name of the new process requested.
    process::name_t const m_new_process;
    /// The name of the new port requested.
    process::port_t const m_new_port;
};

/**
 * \class reset_running_pipeline_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline is reset while it is running.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT reset_running_pipeline_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    reset_running_pipeline_exception() throw();
    /**
     * \brief Destructor.
     */
    ~reset_running_pipeline_exception() throw();
};

/**
 * \class pipeline_not_setup_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline that has not been setup is started.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_not_setup_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_not_setup_exception() throw();
    /**
     * \brief Destructor.
     */
    ~pipeline_not_setup_exception() throw();
};

/**
 * \class pipeline_not_ready_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline that did not succeed in setting up is started.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_not_ready_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_not_ready_exception() throw();
    /**
     * \brief Destructor.
     */
    ~pipeline_not_ready_exception() throw();
};

/**
 * \class pipeline_not_running_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when a pipeline that is not running is stopped.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_not_running_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipeline_not_running_exception() throw();
    /**
     * \brief Destructor.
     */
    ~pipeline_not_running_exception() throw();
};

}

#endif // VISTK_PIPELINE_PIPELINE_EXCEPTION_H
