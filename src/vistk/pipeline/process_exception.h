/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_EXCEPTION_H
#define VISTK_PIPELINE_PROCESS_EXCEPTION_H

#include "pipeline-config.h"

#include "process.h"
#include "types.h"

#include <string>

namespace vistk
{

/**
 * \class process_exception
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
 * \class port_connection_exception
 *
 * \brief The base class used when an error occurs when connecting to a port.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT port_connection_exception
  : public process_exception
{
  public:
    port_connection_exception(process::name_t const& process, process::port_t const& port) throw();
    virtual ~port_connection_exception() throw();

    /// The name of the \ref process which was connected to.
    process::name_t const m_process;
    /// The name of the port which was connected to.
    process::port_t const m_port;
};

/**
 * \class no_such_port_exception
 *
 * \brief Thrown when a connection to a port that does not exist is requested.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_port_exception
  : public port_connection_exception
{
  public:
    no_such_port_exception(process::name_t const& process, process::port_t const& port) throw();
    ~no_such_port_exception() throw();

    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class null_edge_port_connection
 *
 * \brief Thrown when a connection to a port is given an \ref edge that is \c NULL.
 */
class VISTK_PIPELINE_EXPORT null_edge_port_connection
  : public port_connection_exception
{
  public:
    null_edge_port_connection(process::name_t const& process, process::port_t const& port) throw();
    ~null_edge_port_connection() throw();

    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class port_reconnect_exception
 *
 * \brief Thrown when a port that is already connected is connected to again.
 */
class VISTK_PIPELINE_EXPORT port_reconnect_exception
  : public port_connection_exception
{
  public:
    port_reconnect_exception(process::name_t const& process, process::port_t const& port) throw();
    ~port_reconnect_exception() throw();

    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class process_configuration_exception
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
 * \class unknown_configuration_value
 *
 * \brief Thrown when a requested configuration value does not exist.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT unknown_configuration_value
  : public process_configuration_exception
{
  public:
    unknown_configuration_value(process::name_t const& process, config::key_t const& key) throw();
    ~unknown_configuration_value() throw();

    /// The name of the \ref process which was connected to.
    process::name_t const m_process;
    /// The name of the key which was given.
    config::key_t const m_key;

    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class invalid_configuration_value
 *
 * \brief Thrown when a configuration value has an invalid value.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT invalid_configuration_value
  : public process_configuration_exception
{
  public:
    invalid_configuration_value(process::name_t const& process, config::key_t const& key, config::value_t const& value, config::description_t const& desc) throw();
    ~invalid_configuration_value() throw();

    /// The name of the \ref process which was connected to.
    process::name_t const m_process;
    /// The name of the key which was given.
    config::key_t const m_key;
    /// The invalid value.
    config::value_t const m_value;
    /// A description of the key.
    config::description_t const m_desc;

    char const* what() const throw();
  private:
    std::string m_what;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_PROCESS_EXCEPTION_H
