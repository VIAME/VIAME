/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_EDGE_EXCEPTION_H
#define VISTK_PIPELINE_EDGE_EXCEPTION_H

#include "edge.h"
#include "process.h"
#include "types.h"

#include <string>

namespace vistk
{

/**
 * \class edge_exception
 *
 * \brief The base class for all exceptions thrown from an \ref edge.
 *
 * \ingroup exceptions
 */
class edge_exception
  : public pipeline_exception
{
};

/**
 * \class edge_connection_exception
 *
 * \brief The base class for all exceptions thrown from an \ref edge due to connections.
 *
 * \ingroup exceptions
 */
class edge_connection_exception
  : public edge_exception
{
};

/**
 * \class null_process_connection
 *
 * \brief Thrown when a \c NULL is given to connect to an \ref edge.
 *
 * \ingroup exceptions
 */
class null_process_connection
  : public edge_connection_exception
{
  public:
    null_process_connection() throw();
    ~null_process_connection() throw();

    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class duplicate_edge_connection_exception
 *
 * \brief Thrown when an \ref edge is given a second \ref process to connect to the \ref edge.
 *
 * \ingroup exceptions
 */
class duplicate_edge_connection_exception
  : public edge_connection_exception
{
  public:
    duplicate_edge_connection_exception(process::name_t const& process, process::name_t const& new_process, std::string const& type) throw();
    virtual ~duplicate_edge_connection_exception() throw();

    char const* what() const throw();

    /// The name of the process which was already connected.
    process::name_t const m_process;
    /// The name of the process which was attempted to be connected.
    process::name_t const m_new_process;
  private:
    std::string m_what;
};

/**
 * \class input_already_connected
 *
 * \brief Thrown when an \ref edge already has an input process set.
 *
 * \ingroup exceptions
 */
class input_already_connected
  : public duplicate_edge_connection_exception
{
  public:
    input_already_connected(process::name_t const& process, process::name_t const& new_process) throw();
    ~input_already_connected() throw();
};

/**
 * \class output_already_connected
 *
 * \brief Thrown when an \ref edge already has an output process set.
 *
 * \ingroup exceptions
 */
class output_already_connected
  : public duplicate_edge_connection_exception
{
  public:
    output_already_connected(process::name_t const& process, process::name_t const& new_process) throw();
    ~output_already_connected() throw();
};

} // end namespace vistk

#endif // VISTK_PIPELINE_EDGE_EXCEPTION_H
