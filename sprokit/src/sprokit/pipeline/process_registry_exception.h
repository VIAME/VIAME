// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H
#define SPROKIT_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "process.h"
#include "types.h"

/**
 * \file process_registry_exception.h
 *
 * \brief Header for exceptions used within the \link sprokit::process_registry process registry\endlink.
 */

namespace sprokit
{

/**
 * \class process_registry_exception process_registry_exception.h <sprokit/pipeline/process_registry_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref process_registry.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT process_registry_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    process_registry_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~process_registry_exception() noexcept;
};

/**
 * \class null_process_ctor_exception process_registry_exception.h <sprokit/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a \c NULL constructor function is added to the registry.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_process_ctor_exception
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type the ctor is for.
     */
    null_process_ctor_exception(process::type_t const& type) noexcept;
    /**
     * \brief Destructor.
     */
    ~null_process_ctor_exception() noexcept;

    /// The type that was passed a \c NULL constructor.
    process::type_t const m_type;
};

/**
 * \class null_process_registry_config_exception process_registry_exception.h <sprokit/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a process.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_process_registry_config_exception
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_process_registry_config_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~null_process_registry_config_exception() noexcept;
};

/**
 * \class no_such_process_type_exception process_registry_exception.h <sprokit/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT no_such_process_type_exception
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    no_such_process_type_exception(process::type_t const& type) noexcept;
    /**
     * \brief Destructor.
     */
    ~no_such_process_type_exception() noexcept;

    /// The type that was requested from the \link process_registry process registry\endlink.
    process::type_t const m_type;
};

/**
 * \class process_type_already_exists_exception process_registry_exception.h <sprokit/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT process_type_already_exists_exception
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    process_type_already_exists_exception(process::type_t const& type) noexcept;
    /**
     * \brief Destructor.
     */
    ~process_type_already_exists_exception() noexcept;

    /// The type that was requested from the \link process_registry process registry\endlink.
    process::type_t const m_type;
};

}

#endif // SPROKIT_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H
