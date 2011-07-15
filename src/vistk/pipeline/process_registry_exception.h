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

namespace vistk
{

/**
 * \class process_registry_exception
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
 * \class no_such_process_type
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_process_type
  : public process_registry_exception
{
  public:
    no_such_process_type(process_registry::type_t const& type) throw();
    ~no_such_process_type() throw();

    /// The type that was requested from the \link process_registry process registry\endlink.
    process_registry::type_t const m_type;

    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class process_type_already_exists
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT process_type_already_exists
  : public process_registry_exception
{
  public:
    process_type_already_exists(process_registry::type_t const& type) throw();
    ~process_type_already_exists() throw();

    /// The type that was requested from the \link process_registry process registry\endlink.
    process_registry::type_t const m_type;

    char const* what() const throw();
  private:
    std::string m_what;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_PROCESS_REGISTRY_EXCEPTION_H
