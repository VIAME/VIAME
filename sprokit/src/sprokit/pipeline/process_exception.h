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

#ifndef SPROKIT_PIPELINE_PROCESS_EXCEPTION_H
#define SPROKIT_PIPELINE_PROCESS_EXCEPTION_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include <vital/config/config_block.h>
#include "process.h"
#include "types.h"

#include <string>

/**
 * \file process_exception.h
 *
 * \brief Header for exceptions used within \link sprokit::process processes\endlink.
 */

namespace sprokit {

// ----------------------------------------------------------------------------
/**
 * \class process_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref process.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT process_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    process_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~process_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class null_process_config_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a process.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_process_config_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_process_config_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~null_process_config_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class already_initialized_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a process has been initialized before configure.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT already_initialized_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     */
    already_initialized_exception(process::name_t const& name) noexcept;
    /**
     * \brief Destructor.
     */
    ~already_initialized_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
};

// ----------------------------------------------------------------------------
/**
 * \class unconfigured_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a process hasn't been configured before initialization or stepping.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT unconfigured_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     */
    unconfigured_exception(process::name_t const& name) noexcept;
    /**
     * \brief Destructor.
     */
    ~unconfigured_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
};

// ----------------------------------------------------------------------------
/**
 * \class reconfigured_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a process is configured for a second time.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT reconfigured_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     */
    reconfigured_exception(process::name_t const& name) noexcept;
    /**
     * \brief Destructor.
     */
    ~reconfigured_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
};

// ----------------------------------------------------------------------------
/**
 * \class reinitialization_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a process is initialized for a second time.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT reinitialization_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     */
    reinitialization_exception(process::name_t const& name) noexcept;
    /**
     * \brief Destructor.
     */
    ~reinitialization_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
};

// ----------------------------------------------------------------------------
/**
 * \class null_conf_info_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a port is declared with a \c NULL info structure.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_conf_info_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param key The configuration key with \c NULL information.
     */
    null_conf_info_exception(process::name_t const& name, kwiver::vital::config_block_key_t const& key) noexcept;
    /**
     * \brief Destructor.
     */
    ~null_conf_info_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
    /// The configuration key.
    kwiver::vital::config_block_key_t const m_key;
};

// ----------------------------------------------------------------------------
/**
 * \class null_port_info_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when \c NULL is passed as information for a port.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_port_info_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The port with \c NULL information.
     * \param type The type of port.
     */
    null_port_info_exception(process::name_t const& name, process::port_t const& port, std::string const& type) noexcept;
    /**
     * \brief Destructor.
     */
    ~null_port_info_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
    /// The name of the port.
    process::port_t const m_port;
};

// ----------------------------------------------------------------------------
/**
 * \class null_input_port_info_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when \c NULL is passed as information for an input port.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_input_port_info_exception
  : public null_port_info_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the \ref process.
     * \param port The port with \c NULL information.
     */
    null_input_port_info_exception(process::name_t const& name, process::port_t const& port) noexcept;
    /**
     * \brief Destructor.
     */
    ~null_input_port_info_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class null_output_port_info_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when \c NULL is passed as information for an output port.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_output_port_info_exception
  : public null_port_info_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the \ref process.
     * \param port The port with \c NULL information.
     */
    null_output_port_info_exception(process::name_t const& name, process::port_t const& port) noexcept;
    /**
     * \brief Destructor.
     */
    ~null_output_port_info_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class flag_mismatch_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when incompatible flags are given for a port.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT flag_mismatch_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the \ref process.
     * \param port The port with \c NULL information.
     * \param reason The reason why the flags are incompatible.
     */
    flag_mismatch_exception(process::name_t const& name, process::port_t const& port, std::string const& reason) noexcept;
    /**
     * \brief Destructor.
     */
    ~flag_mismatch_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
    /// The name of the port.
    process::port_t const m_port;
    /// A reason for the incompatible flags.
    std::string const m_reason;
};

// ----------------------------------------------------------------------------
/**
 * \class set_type_on_initialized_process_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when the type on a port is attempted to be set after initialization.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT set_type_on_initialized_process_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the \ref process.
     * \param port The name of the port on the \ref process.
     * \param type The type that was attempted to be set.
     */
    set_type_on_initialized_process_exception(process::name_t const& name, process::port_t const& port, process::port_type_t const& type) noexcept;
    /**
     * \brief Destructor.
     */
    ~set_type_on_initialized_process_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
    /// The name of the port.
    process::port_t const m_port;
    /// The type that was attempted to be set.
    process::port_type_t const m_type;
};

// ----------------------------------------------------------------------------
/**
 * \class set_frequency_on_initialized_process_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when the frequency on a port is attempted to be set after initialization.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT set_frequency_on_initialized_process_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the \ref process.
     * \param port The name of the port on the \ref process.
     * \param frequency The frequency that was attempted to be set.
     */
    set_frequency_on_initialized_process_exception(process::name_t const& name,
                                                   process::port_t const& port,
                                                   process::port_frequency_t const& frequency) noexcept;
    /**
     * \brief Destructor.
     */
    ~set_frequency_on_initialized_process_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
    /// The name of the port.
    process::port_t const m_port;
    /// The frequency that was attempted to be set.
    process::port_frequency_t const m_frequency;
};

// ----------------------------------------------------------------------------
/**
 * \class uninitialized_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a process is stepped before initialization.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT uninitialized_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     */
    uninitialized_exception(process::name_t const& name) noexcept;
    /**
     * \brief Destructor.
     */
    ~uninitialized_exception() noexcept;

    /// The name of the \ref process.
    process::name_t const m_name;
};

// ----------------------------------------------------------------------------
/**
 * \class port_connection_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief The base class used when an error occurs when connecting to a port.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT port_connection_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the port.
     */
    port_connection_exception(process::name_t const& name, process::port_t const& port) noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~port_connection_exception() noexcept;

    /// The name of the \ref process which was connected to.
    process::name_t const m_name;
    /// The name of the port which was connected to.
    process::port_t const m_port;
};

// ----------------------------------------------------------------------------
/**
 * \class connect_to_initialized_process_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a connection is requested to be made to an initialized process.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT connect_to_initialized_process_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the port.
     */
    connect_to_initialized_process_exception(process::name_t const& name, process::port_t const& port) noexcept;
    /**
     * \brief Destructor.
     */
    ~connect_to_initialized_process_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class no_such_port_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a connection to a port that does not exist is requested.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT no_such_port_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the port.
     */
    no_such_port_exception(process::name_t const& name, process::port_t const& port) noexcept;

    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the port.
     * \param all_ports List of all available ports
     */
    no_such_port_exception(process::name_t const& name, process::port_t const& port,
                                        process::ports_t const& all_ports) noexcept;


    /**
     * \brief Destructor.
     */
    ~no_such_port_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class null_edge_port_connection_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a connection to a port is given an \ref edge that is \c NULL.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_edge_port_connection_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the port.
     */
    null_edge_port_connection_exception(process::name_t const& name, process::port_t const& port) noexcept;

    /**
     * \brief Destructor.
     */
    ~null_edge_port_connection_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class static_type_reset_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a port type is attempted to be reset on a static type.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT static_type_reset_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the port.
     * \param orig_type The original type of the port.
     * \param new_type The type that was attempted to be set on the port.
     */
    static_type_reset_exception(process::name_t const& name, process::port_t const& port, process::port_type_t const& orig_type, process::port_type_t const& new_type) noexcept;
    /**
     * \brief Destructor.
     */
    ~static_type_reset_exception() noexcept;

    /// The original type on the port.
    process::port_type_t const m_orig_type;
    /// The new type for the port.
    process::port_type_t const m_new_type;
};

// ----------------------------------------------------------------------------
/**
 * \class port_reconnect_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a port that is already connected is connected to again.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT port_reconnect_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the port.
     */
    port_reconnect_exception(process::name_t const& name, process::port_t const& port) noexcept;
    /**
     * \brief Destructor.
     */
    ~port_reconnect_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class missing_connection_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a connection to a port that is marked as required is missing.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT missing_connection_exception
  : public port_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param port The name of the port.
     * \param reason The reason why the connection is necessary.
     */
    missing_connection_exception(process::name_t const& name, process::port_t const& port, std::string const& reason) noexcept;
    /**
     * \brief Destructor.
     */
    ~missing_connection_exception() noexcept;

    /// A reason for the missing connection.
    std::string const m_reason;
};

// ----------------------------------------------------------------------------
/**
 * \class process_configuration_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a \ref process has a configuration issue.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT process_configuration_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     */
    process_configuration_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~process_configuration_exception() noexcept;
};

// ----------------------------------------------------------------------------
/**
 * \class unknown_configuration_value_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a requested configuration value does not exist.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT unknown_configuration_value_exception
  : public process_configuration_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param key The key requested.
     */
    unknown_configuration_value_exception(process::name_t const& name, kwiver::vital::config_block_key_t const& key) noexcept;
    /**
     * \brief Destructor.
     */
    ~unknown_configuration_value_exception() noexcept;

    /// The name of the \ref process which was connected to.
    process::name_t const m_name;
    /// The name of the key which was given.
    kwiver::vital::config_block_key_t const m_key;
};

// ----------------------------------------------------------------------------
/**
 * \class invalid_configuration_value_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a configuration value has an invalid value.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT invalid_configuration_value_exception
  : public process_configuration_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param key The key requested.
     * \param value The value given.
     * \param desc A description of the configuration value.
     */
    invalid_configuration_value_exception(process::name_t const& name, kwiver::vital::config_block_key_t const& key, kwiver::vital::config_block_value_t const& value, kwiver::vital::config_block_description_t const& desc) noexcept;
    /**
     * \brief Destructor.
     */
    ~invalid_configuration_value_exception() noexcept;

    /// The name of the \ref process which was connected to.
    process::name_t const m_name;
    /// The name of the key which was given.
    kwiver::vital::config_block_key_t const m_key;
    /// The invalid value.
    kwiver::vital::config_block_value_t const m_value;
    /// A description of the key.
    kwiver::vital::config_block_description_t const m_desc;
};

// ----------------------------------------------------------------------------
/**
 * \class invalid_configuration_exception process_exception.h <sprokit/pipeline/process_exception.h>
 *
 * \brief Thrown when a configuration for a \ref process is invalid.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT invalid_configuration_exception
  : public process_configuration_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param reason The reason why the configuration is invalid.
     */
    invalid_configuration_exception(process::name_t const& name, std::string const& reason) noexcept;
    /**
     * \brief Destructor.
     */
    ~invalid_configuration_exception() noexcept;

    /// The name of the \ref process which was connected to.
    process::name_t const m_name;
    /// A reason for the invalid configuration.
    std::string const m_reason;
};

}

#endif // SPROKIT_PIPELINE_PROCESS_EXCEPTION_H
