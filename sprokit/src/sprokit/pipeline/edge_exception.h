// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_EDGE_EXCEPTION_H
#define SPROKIT_PIPELINE_EDGE_EXCEPTION_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "process.h"
#include "types.h"

#include <string>

/**
 * \file edge_exception.h
 *
 * \brief Header for exceptions used within \link sprokit::edge edges\endlink.
 */

namespace sprokit
{

/**
 * \class edge_exception edge_exception.h <sprokit/pipeline/edge_exception.h>
 *
 * \brief The base class for all exceptions thrown from an \ref edge.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT edge_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    edge_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~edge_exception() noexcept;
};

/**
 * \class null_edge_config_exception pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when \c NULL \ref config is passed to a edge.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_edge_config_exception
  : public edge_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_edge_config_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~null_edge_config_exception() noexcept;
};

/**
 * \class datum_requested_after_complete pipeline_exception.h <sprokit/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when data was requested after completion was indicated.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT datum_requested_after_complete
  : public edge_exception
{
  public:
    /**
     * \brief Constructor.
     */
    datum_requested_after_complete() noexcept;
    /**
     * \brief Destructor.
     */
    ~datum_requested_after_complete() noexcept;
};

/**
 * \class edge_connection_exception edge_exception.h <sprokit/pipeline/edge_exception.h>
 *
 * \brief The base class for all exceptions thrown from an \ref edge due to connections.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT edge_connection_exception
  : public edge_exception
{
  public:
    /**
     * \brief Constructor.
     */
    edge_connection_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~edge_connection_exception() noexcept;
};

/**
 * \class null_process_connection_exception edge_exception.h <sprokit/pipeline/edge_exception.h>
 *
 * \brief Thrown when a \c NULL is given to connect to an \ref edge.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_process_connection_exception
  : public edge_connection_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_process_connection_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~null_process_connection_exception() noexcept;
};

/**
 * \class duplicate_edge_connection_exception edge_exception.h <sprokit/pipeline/edge_exception.h>
 *
 * \brief Thrown when an \ref edge is given a second \ref process to connect to the \ref edge.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT duplicate_edge_connection_exception
  : public edge_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process that was already connected.
     * \param new_name The name of the process which was attemted to be connected.
     * \param type The type of connection.
     */
    duplicate_edge_connection_exception(process::name_t const& name, process::name_t const& new_name, std::string const& type) noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~duplicate_edge_connection_exception() noexcept;

    /// The name of the process which was already connected.
    process::name_t const m_name;
    /// The name of the process which was attempted to be connected.
    process::name_t const m_new_name;
};

/**
 * \class input_already_connected_exception edge_exception.h <sprokit/pipeline/edge_exception.h>
 *
 * \brief Thrown when an \ref edge already has an input process set.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT input_already_connected_exception
  : public duplicate_edge_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process that was already connected.
     * \param new_name The name of the process which was attemted to be connected.
     */
    input_already_connected_exception(process::name_t const& name, process::name_t const& new_name) noexcept;
    /**
     * \brief Destructor.
     */
    ~input_already_connected_exception() noexcept;
};

/**
 * \class output_already_connected_exception edge_exception.h <sprokit/pipeline/edge_exception.h>
 *
 * \brief Thrown when an \ref edge already has an output process set.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT output_already_connected_exception
  : public duplicate_edge_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process that was already connected.
     * \param new_name The name of the process which was attemted to be connected.
     */
    output_already_connected_exception(process::name_t const& name, process::name_t const& new_name) noexcept;
    /**
     * \brief Destructor.
     */
    ~output_already_connected_exception() noexcept;
};

}

#endif // SPROKIT_PIPELINE_EDGE_EXCEPTION_H
