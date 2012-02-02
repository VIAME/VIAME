/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_EXCEPTION_H
#define VISTK_PIPELINE_PROCESS_EXCEPTION_H

#include "pipeline-config.h"

#include "config.h"
#include "process.h"
#include "types.h"

#include <string>

/**
 * \file process_exception.h
 *
 * \brief Header for exceptions used within \link vistk::process processes\endlink.
 */

namespace vistk
{

/**
 * \class process_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref process.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT process_exception
  : public pipeline_exception
{
};

/**
 * \class null_process_config_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when \c NULL \ref config is passed to a process.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_process_config_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_process_config_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_process_config_exception() throw();
};

/**
 * \class reinitialization_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a process is initialized for a second time.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT reinitialization_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     */
    reinitialization_exception(process::name_t const& process) throw();
    /**
     * \brief Destructor.
     */
    ~reinitialization_exception() throw();

    /// The name of the \ref process.
    process::name_t const m_process;
};

/**
 * \class null_conf_info_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a port is declared with a \c NULL info structure.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_conf_info_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param key The configuration key with \c NULL information.
     */
    null_conf_info_exception(process::name_t const& process, config::key_t const& key) throw();
    /**
     * \brief Destructor.
     */
    ~null_conf_info_exception() throw();

    /// The name of the \ref process.
    process::name_t const m_process;
    /// The configuration key.
    config::key_t const m_key;
};

/**
 * \class null_port_info_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when \c NULL is passed as information for a port.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_port_info_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The port with \c NULL information.
     * \param type The type of port.
     */
    null_port_info_exception(process::name_t const& process, process::port_t const& port, std::string const& type) throw();
    /**
     * \brief Destructor.
     */
    ~null_port_info_exception() throw();

    /// The name of the \ref process.
    process::name_t const m_process;
    /// The name of the port.
    process::port_t const m_port;
};

/**
 * \class null_input_port_info_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when \c NULL is passed as information for an input port.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_input_port_info_exception
  : public null_port_info_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The port with \c NULL information.
     */
    null_input_port_info_exception(process::name_t const& process, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    ~null_input_port_info_exception() throw();

    /// The name of the \ref process.
    process::name_t const m_process;
    /// The name of the port.
    process::port_t const m_port;
};

/**
 * \class null_output_port_info_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when \c NULL is passed as information for an output port.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_output_port_info_exception
  : public null_port_info_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The port with \c NULL information.
     */
    null_output_port_info_exception(process::name_t const& process, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    ~null_output_port_info_exception() throw();

    /// The name of the \ref process.
    process::name_t const m_process;
    /// The name of the port.
    process::port_t const m_port;
};

/**
 * \class set_type_on_initialized_process_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when \c NULL is passed as information for an output port.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT set_type_on_initialized_process_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the \ref process.
     * \param port The name of the port on the \ref process.
     * \param type The type that was attempted to be set.
     */
    set_type_on_initialized_process_exception(process::name_t const& process, process::port_t const& port, process::port_type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~set_type_on_initialized_process_exception() throw();

    /// The name of the \ref process.
    process::name_t const m_process;
    /// The name of the port.
    process::port_t const m_port;
    /// The type that was attempted to be set.
    process::port_type_t const m_type;
};

/**
 * \class uninitialized_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a process is stepped before initialization.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT uninitialized_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     */
    uninitialized_exception(process::name_t const& process) throw();
    /**
     * \brief Destructor.
     */
    ~uninitialized_exception() throw();

    /// The name of the \ref process.
    process::name_t const m_process;
};

/**
 * \class port_connection_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief The base class used when an error occurs when connecting to a port.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT port_connection_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The name of the port.
     */
    port_connection_exception(process::name_t const& process, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    virtual ~port_connection_exception() throw();

    /// The name of the \ref process which was connected to.
    process::name_t const m_process;
    /// The name of the port which was connected to.
    process::port_t const m_port;
};

/**
 * \class connect_to_initialized_process_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a connection is requested to be made to an initialized process.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT connect_to_initialized_process_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The name of the port.
     */
    connect_to_initialized_process_exception(process::name_t const& process, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    ~connect_to_initialized_process_exception() throw();
};

/**
 * \class no_such_port_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a connection to a port that does not exist is requested.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_port_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The name of the port.
     */
    no_such_port_exception(process::name_t const& process, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_port_exception() throw();
};

/**
 * \class null_edge_port_connection_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a connection to a port is given an \ref edge that is \c NULL.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_edge_port_connection_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The name of the port.
     */
    null_edge_port_connection_exception(process::name_t const& process, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    ~null_edge_port_connection_exception() throw();
};

/**
 * \class static_type_reset_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a port type is attempted to be reset on a static type.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT static_type_reset_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The name of the port.
     * \param orig_type The original type of the port.
     * \param new_type The type that was attempted to be set on the port.
     */
    static_type_reset_exception(process::name_t const& process, process::port_t const& port, process::port_type_t const& orig_type, process::port_type_t const& new_type) throw();
    /**
     * \brief Destructor.
     */
    ~static_type_reset_exception() throw();

    process::port_type_t const m_orig_type;
    process::port_type_t const m_new_type;
};

/**
 * \class port_reconnect_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a port that is already connected is connected to again.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT port_reconnect_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The name of the port.
     */
    port_reconnect_exception(process::name_t const& process, process::port_t const& port) throw();
    /**
     * \brief Destructor.
     */
    ~port_reconnect_exception() throw();
};

/**
 * \class missing_connection_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a connection to a port that is necessary is missing.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT missing_connection_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param port The name of the port.
     * \param reason The reason why the connection is necessary.
     */
    missing_connection_exception(process::name_t const& process, process::port_t const& port, std::string const& reason) throw();
    /**
     * \brief Destructor.
     */
    ~missing_connection_exception() throw();

    /// A reason for the missing connection.
    std::string const m_reason;
};

/**
 * \class process_configuration_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a \ref process has a configuration issue.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT process_configuration_exception
  : public process_exception
{
};

/**
 * \class unknown_configuration_value_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a requested configuration value does not exist.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT unknown_configuration_value_exception
  : public process_configuration_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param key The key requested.
     */
    unknown_configuration_value_exception(process::name_t const& process, config::key_t const& key) throw();
    /**
     * \brief Destructor.
     */
    ~unknown_configuration_value_exception() throw();

    /// The name of the \ref process which was connected to.
    process::name_t const m_process;
    /// The name of the key which was given.
    config::key_t const m_key;
};

/**
 * \class invalid_configuration_value_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a configuration value has an invalid value.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT invalid_configuration_value_exception
  : public process_configuration_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param key The key requested.
     * \param value The value given.
     * \param desc A description of the configuration value.
     */
    invalid_configuration_value_exception(process::name_t const& process, config::key_t const& key, config::value_t const& value, config::description_t const& desc) throw();
    /**
     * \brief Destructor.
     */
    ~invalid_configuration_value_exception() throw();

    /// The name of the \ref process which was connected to.
    process::name_t const m_process;
    /// The name of the key which was given.
    config::key_t const m_key;
    /// The invalid value.
    config::value_t const m_value;
    /// A description of the key.
    config::description_t const m_desc;
};

/**
 * \class invalid_configuration_exception process_exception.h <vistk/pipeline/process_exception.h>
 *
 * \brief Thrown when a configuration for a \ref process is invalid.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT invalid_configuration_exception
  : public process_configuration_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process.
     * \param reason The reason why the configuration is invalid.
     */
    invalid_configuration_exception(process::name_t const& process, std::string const& reason) throw();
    /**
     * \brief Destructor.
     */
    ~invalid_configuration_exception() throw();

    /// The name of the \ref process which was connected to.
    process::name_t const m_process;
    /// A reason for the invalid configuration.
    std::string const m_reason;
};

}

#endif // VISTK_PIPELINE_PROCESS_EXCEPTION_H
