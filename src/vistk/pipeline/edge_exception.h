/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_EDGE_EXCEPTION_H
#define VISTK_PIPELINE_EDGE_EXCEPTION_H

#include "pipeline-config.h"

#include "process.h"
#include "types.h"

#include <string>

/**
 * \file edge_exception.h
 *
 * \brief Header for exceptions used within \link vistk::edge edges\endlink.
 */

namespace vistk
{

/**
 * \class edge_exception edge_exception.h <vistk/pipeline/edge_exception.h>
 *
 * \brief The base class for all exceptions thrown from an \ref edge.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT edge_exception
  : public pipeline_exception
{
};

/**
 * \class null_edge_config_exception pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when \c NULL \ref config is passed to a edge.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_edge_config_exception
  : public edge_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_edge_config_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_edge_config_exception() throw();
};

/**
 * \class datum_requested_after_complete pipeline_exception.h <vistk/pipeline/pipeline_exception.h>
 *
 * \brief Thrown when data was requested after completion was indicated.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT datum_requested_after_complete
  : public edge_exception
{
  public:
    /**
     * \brief Constructor.
     */
    datum_requested_after_complete() throw();
    /**
     * \brief Destructor.
     */
    ~datum_requested_after_complete() throw();
};

/**
 * \class edge_connection_exception edge_exception.h <vistk/pipeline/edge_exception.h>
 *
 * \brief The base class for all exceptions thrown from an \ref edge due to connections.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT edge_connection_exception
  : public edge_exception
{
};

/**
 * \class null_process_connection_exception edge_exception.h <vistk/pipeline/edge_exception.h>
 *
 * \brief Thrown when a \c NULL is given to connect to an \ref edge.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_process_connection_exception
  : public edge_connection_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_process_connection_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_process_connection_exception() throw();
};

/**
 * \class duplicate_edge_connection_exception edge_exception.h <vistk/pipeline/edge_exception.h>
 *
 * \brief Thrown when an \ref edge is given a second \ref process to connect to the \ref edge.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT duplicate_edge_connection_exception
  : public edge_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process that was already connected.
     * \param new_process The name of the process which was attemted to be connected.
     * \param type The type of connection.
     */
    duplicate_edge_connection_exception(process::name_t const& process, process::name_t const& new_process, std::string const& type) throw();
    /**
     * \brief Destructor.
     */
    virtual ~duplicate_edge_connection_exception() throw();

    /// The name of the process which was already connected.
    process::name_t const m_process;
    /// The name of the process which was attempted to be connected.
    process::name_t const m_new_process;
};

/**
 * \class input_already_connected_exception edge_exception.h <vistk/pipeline/edge_exception.h>
 *
 * \brief Thrown when an \ref edge already has an input process set.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT input_already_connected_exception
  : public duplicate_edge_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process that was already connected.
     * \param new_process The name of the process which was attemted to be connected.
     */
    input_already_connected_exception(process::name_t const& process, process::name_t const& new_process) throw();
    /**
     * \brief Destructor.
     */
    ~input_already_connected_exception() throw();
};

/**
 * \class output_already_connected_exception edge_exception.h <vistk/pipeline/edge_exception.h>
 *
 * \brief Thrown when an \ref edge already has an output process set.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT output_already_connected_exception
  : public duplicate_edge_connection_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param process The name of the process that was already connected.
     * \param new_process The name of the process which was attemted to be connected.
     */
    output_already_connected_exception(process::name_t const& process, process::name_t const& new_process) throw();
    /**
     * \brief Destructor.
     */
    ~output_already_connected_exception() throw();
};

}

#endif // VISTK_PIPELINE_EDGE_EXCEPTION_H
