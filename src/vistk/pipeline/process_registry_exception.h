/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H
#define VISTK_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H

#include "pipeline-config.h"

#include "process.h"
#include "types.h"

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
 * \class null_process_ctor_exception process_registry_exception.h <vistk/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a \c NULL constructor function is added to the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_process_ctor_exception
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type the ctor is for.
     */
    null_process_ctor_exception(process::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~null_process_ctor_exception() throw();

    /// The type that was passed a \c NULL constructor.
    process::type_t const m_type;
};

/**
 * \class null_process_registry_config_exception process_registry_exception.h <vistk/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a process.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_process_registry_config_exception
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_process_registry_config_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_process_registry_config_exception() throw();
};

/**
 * \class no_such_process_type_exception process_registry_exception.h <vistk/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_process_type_exception
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    no_such_process_type_exception(process::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_process_type_exception() throw();

    /// The type that was requested from the \link process_registry process registry\endlink.
    process::type_t const m_type;
};

/**
 * \class process_type_already_exists_exception process_registry_exception.h <vistk/pipeline/process_registry_exception.h>
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT process_type_already_exists_exception
  : public process_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    process_type_already_exists_exception(process::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~process_type_already_exists_exception() throw();

    /// The type that was requested from the \link process_registry process registry\endlink.
    process::type_t const m_type;
};

}

#endif // VISTK_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H
