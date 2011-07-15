/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_EDGE_REGISTRY_EXCEPTION_H
#define VISTK_PIPELINE_EDGE_REGISTRY_EXCEPTION_H

#include "pipeline-config.h"

#include "edge_registry.h"
#include "types.h"

#include <string>

namespace vistk
{

/**
 * \class edge_registry_exception
 *
 * \brief The base class for all exceptions thrown from an \ref edge_registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT edge_registry_exception
  : public pipeline_exception
{
};

/**
 * \class no_such_edge_type
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_edge_type
  : public edge_registry_exception
{
  public:
    no_such_edge_type(edge_registry::type_t const& type) throw();
    ~no_such_edge_type() throw();

    /// The type that was requested from the \link edge_registry edge registry\endlink.
    edge_registry::type_t const m_type;

    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class edge_type_already_exists
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT edge_type_already_exists
  : public edge_registry_exception
{
  public:
    edge_type_already_exists(edge_registry::type_t const& type) throw();
    ~edge_type_already_exists() throw();

    /// The type that was requested for the \link edge_registry edge registry\endlink.
    edge_registry::type_t const m_type;

    char const* what() const throw();
  private:
    std::string m_what;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_EDGE_REGISTRY_EXCEPTION_H
