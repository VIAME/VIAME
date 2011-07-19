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

/**
 * \file edge_registry_exception.h
 *
 * \brief Header for exceptions used within the \link edge_registry edge registry\endlink.
 */

namespace vistk
{

/**
 * \class edge_registry_exception edge_registry_exception.h <vistk/pipeline/edge_registry_exception.h>
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
 * \class no_such_edge_type edge_registry_exception.h <vistk/pipeline/edge_registry_exception.h>
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_edge_type
  : public edge_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    no_such_edge_type(edge_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_edge_type() throw();

    /// The type that was requested from the \link edge_registry edge registry\endlink.
    edge_registry::type_t const m_type;

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
 * \class edge_type_already_exists edge_registry_exception.h <vistk/pipeline/edge_registry_exception.h>
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT edge_type_already_exists
  : public edge_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    edge_type_already_exists(edge_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~edge_type_already_exists() throw();

    /// The type that was requested for the \link edge_registry edge registry\endlink.
    edge_registry::type_t const m_type;

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

#endif // VISTK_PIPELINE_EDGE_REGISTRY_EXCEPTION_H
