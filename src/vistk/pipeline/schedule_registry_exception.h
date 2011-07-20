/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_SCHEDULE_REGISTRY_EXCEPTION_H
#define VISTK_PIPELINE_SCHEDULE_REGISTRY_EXCEPTION_H

#include "pipeline-config.h"

#include "schedule_registry.h"
#include "types.h"

#include <string>

/**
 * \file schedule_registry_exception.h
 *
 * \brief Header for exceptions used within the \link schedule_registry schedule registry\endlink.
 */

namespace vistk
{

/**
 * \class schedule_registry_exception schedule_registry_exception.h <vistk/pipeline/schedule_registry_exception.h>
 *
 * \brief The base class for all exceptions thrown from a \ref schedule_registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT schedule_registry_exception
  : public pipeline_exception
{
};

/**
 * \class no_such_schedule_type schedule_registry_exception.h <vistk/pipeline/schedule_registry_exception.h>
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_schedule_type
  : public schedule_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    no_such_schedule_type(schedule_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_schedule_type() throw();

    /// The type that was requested from the \link schedule_registry schedule registry\endlink.
    schedule_registry::type_t const m_type;

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
 * \class schedule_type_already_exists schedule_registry_exception.h <vistk/pipeline/schedule_registry_exception.h>
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT schedule_type_already_exists
  : public schedule_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    schedule_type_already_exists(schedule_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~schedule_type_already_exists() throw();

    /// The type that was requested from the \link schedule_registry schedule registry\endlink.
    schedule_registry::type_t const m_type;

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

#endif // VISTK_PIPELINE_SCHEDULE_REGISTRY_EXCEPTION_H
