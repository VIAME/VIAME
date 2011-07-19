/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PIPELINE_REGISTRY_EXCEPTION_H
#define VISTK_PIPELINE_PIPELINE_REGISTRY_EXCEPTION_H

#include "pipeline-config.h"

#include "pipeline_registry.h"
#include "types.h"

#include <string>

namespace vistk
{

/**
 * \class pipeline_registry_exception
 *
 * \brief The base class for all exceptions thrown from a \ref pipeline_registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_registry_exception
  : public pipeline_exception
{
};

/**
 * \class no_such_pipeline_type
 *
 * \brief Thrown when a type is requested, but does not exist in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT no_such_pipeline_type
  : public pipeline_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    no_such_pipeline_type(pipeline_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~no_such_pipeline_type() throw();

    /// The type that was requested from the \link pipeline_registry pipeline registry\endlink.
    pipeline_registry::type_t const m_type;

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
 * \class pipeline_type_already_exists
 *
 * \brief Thrown when a type is added, but does already exists in the registry.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_EXPORT pipeline_type_already_exists
  : public pipeline_registry_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param type The type requested.
     */
    pipeline_type_already_exists(pipeline_registry::type_t const& type) throw();
    /**
     * \brief Destructor.
     */
    ~pipeline_type_already_exists() throw();

    /// The type that was requested for the \link pipeline_registry pipeline registry\endlink.
    pipeline_registry::type_t const m_type;

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

#endif // VISTK_PIPELINE_PIPELINE_REGISTRY_EXCEPTION_H
