// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file scheduler_registry_exception.h
 *
 * \brief Header for exceptions used within the \link sprokit::scheduler_registry scheduler registry\endlink.
 */

#ifndef SPROKIT_PIPELINE_SCHEDULER_REGISTRY_EXCEPTION_H
#define SPROKIT_PIPELINE_SCHEDULER_REGISTRY_EXCEPTION_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "pipeline_exception.h"
#include "scheduler.h"
#include "types.h"

namespace sprokit {

/**
 * \class scheduler_registry_exception scheduler_registry_exception.h <sprokit/pipeline/scheduler_registry_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref scheduler_registry.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT scheduler_registry_exception
  : public pipeline_exception
{
public:
  /**
   * \brief Constructor.
   */
  scheduler_registry_exception() throw ( );
  /**
   * \brief Destructor.
   */
  virtual ~scheduler_registry_exception() throw ( );
};

/**
 * \class null_scheduler_ctor_exception scheduler_registry_exception.h <sprokit/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a \c NULL constructor function is added to the registry.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_scheduler_ctor_exception
  : public scheduler_registry_exception
{
public:
  /**
   * \brief Constructor.
   *
   * \param type The type the ctor is for.
   */
  null_scheduler_ctor_exception( sprokit::scheduler::type_t const& type ) throw ( );
  /**
   * \brief Destructor.
   */
  ~null_scheduler_ctor_exception() throw ( );

  /// The type that was passed a \c NULL constructor.
  sprokit::scheduler::type_t const m_type;
};

/**
 * \class null_scheduler_registry_config_exception scheduler_registry_exception.h <sprokit/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a \c NULL \ref config is passed to a scheduler.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_scheduler_registry_config_exception
  : public scheduler_registry_exception
{
public:
  /**
   * \brief Constructor.
   */
  null_scheduler_registry_config_exception() throw ( );
  /**
   * \brief Destructor.
   */
  ~null_scheduler_registry_config_exception() throw ( );
};

/**
 * \class null_scheduler_registry_pipeline_exception scheduler_registry_exception.h <sprokit/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a \c NULL \link sprokit::pipeline\endlink is passed to a scheduler.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT null_scheduler_registry_pipeline_exception
  : public scheduler_registry_exception
{
public:
  /**
   * \brief Constructor.
   */
  null_scheduler_registry_pipeline_exception() throw ( );
  /**
   * \brief Destructor.
   */
  ~null_scheduler_registry_pipeline_exception() throw ( );
};

/**
 * \class no_such_scheduler_type_exception scheduler_registry_exception.h <sprokit/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT no_such_scheduler_type_exception
  : public scheduler_registry_exception
{
public:
  /**
   * \brief Constructor.
   *
   * \param type The type requested.
   */
  no_such_scheduler_type_exception( sprokit::scheduler::type_t const& type ) throw ( );
  /**
   * \brief Destructor.
   */
  ~no_such_scheduler_type_exception() throw ( );

  /// The type that was requested from the \link scheduler_registry scheduler registry\endlink.
  sprokit::scheduler::type_t const m_type;
};

/**
 * \class scheduler_type_already_exists_exception scheduler_registry_exception.h <sprokit/pipeline/scheduler_registry_exception.h>
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT scheduler_type_already_exists_exception
  : public scheduler_registry_exception
{
public:
  /**
   * \brief Constructor.
   *
   * \param type The type requested.
   */
  scheduler_type_already_exists_exception( sprokit::scheduler::type_t const& type ) throw ( );
  /**
   * \brief Destructor.
   */
  ~scheduler_type_already_exists_exception() throw ( );

  /// The type that was requested from the \link scheduler_registry scheduler registry\endlink.
  sprokit::scheduler::type_t const m_type;
};

}

#endif // SPROKIT_PIPELINE_SCHEDULER_REGISTRY_EXCEPTION_H
