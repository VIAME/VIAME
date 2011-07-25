/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H
#define VISTK_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H

#include "pipeline-config.h"

#include "process_registry.h"
#include "types.h"

#include <string>

/**
 * \file process_registry_exception.h
 *
 * \brief Header for exceptions used within the \link vistk::process_registry process registry\endlink.
 */

namespace vistk
{

/**
 * \class process_registry_exception process_registry_exception.h <vistk/pipeline/process_registry_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref process_registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT process_registry_exception
  : public pipeline_exception
{
};

/**
 * \class no_such_process_type process_registry_exception.h <vistk/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_process_type
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    no_such_process_type(process_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_process_type() throw();

    /// The type that was requested from the \link process_registry process registry\endlink.
    process_registry::type_t const m_type;

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class process_type_already_exists process_registry_exception.h <vistk/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT process_type_already_exists
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    process_type_already_exists(process_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~process_type_already_exists() throw();

    /// The type that was requested from the \link process_registry process registry\endlink.
    process_registry::type_t const m_type;

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    char const* what() const throw();
  private:
    std::string m_what;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H
