/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_SCHEDULE_REGISTRY_EXCEPTION_H
#define VISTK_PIPELINE_SCHEDULE_REGISTRY_EXCEPTION_H

#include "pipeline-config.h"

#include "scheduler_registry.h"
#include "types.h"

/**
 * \file scheduler_registry_exception.h
 *
 * \brief Header for exceptions used within the \link vistk::scheduler_registry scheduler registry\endlink.
 */

namespace vistk
{

/**
 * \class scheduler_registry_exception scheduler_registry_exception.h <vistk/pipeline/scheduler_registry_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref scheduler_registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT scheduler_registry_exception
  : public pipeline_exception
{
};

/**
 * \class null_scheduler_ctor_exception scheduler_registry_exception.h <vistk/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a \c NULL constructor function is added to the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_scheduler_ctor_exception
  : public scheduler_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type the ctor is for.
     */
    null_scheduler_ctor_exception(scheduler_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~null_scheduler_ctor_exception() throw();

    /// The type that was passed a \c NULL constructor.
    scheduler_registry::type_t const m_type;
};

/**
 * \class null_scheduler_registry_config_exception scheduler_registry_exception.h <vistk/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a scheduler.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_scheduler_registry_config_exception
  : public scheduler_registry_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_scheduler_registry_config_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_scheduler_registry_config_exception() throw();
};

/**
 * \class null_scheduler_registry_pipeline_exception scheduler_registry_exception.h <vistk/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a \c NULL \link vistk::pipeline\endlink is passed to a scheduler.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT null_scheduler_registry_pipeline_exception
  : public scheduler_registry_exception
{
  public:
    /**
     * \brief Constructor.
     */
    null_scheduler_registry_pipeline_exception() throw();
    /**
     * \brief Destructor.
     */
    ~null_scheduler_registry_pipeline_exception() throw();
};

/**
 * \class no_such_scheduler_type_exception scheduler_registry_exception.h <vistk/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_scheduler_type_exception
  : public scheduler_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    no_such_scheduler_type_exception(scheduler_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_scheduler_type_exception() throw();

    /// The type that was requested from the \link scheduler_registry scheduler registry\endlink.
    scheduler_registry::type_t const m_type;
};

/**
 * \class scheduler_type_already_exists_exception scheduler_registry_exception.h <vistk/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT scheduler_type_already_exists_exception
  : public scheduler_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    scheduler_type_already_exists_exception(scheduler_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~scheduler_type_already_exists_exception() throw();

    /// The type that was requested from the \link scheduler_registry scheduler registry\endlink.
    scheduler_registry::type_t const m_type;
};

}

#endif // VISTK_PIPELINE_SCHEDULE_REGISTRY_EXCEPTION_H
