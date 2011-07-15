/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PIPELINE_REGISTRY_H
#define VISTK_PIPELINE_PIPELINE_REGISTRY_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/function.hpp>
#include <boost/tuple/tuple.hpp>

#include <string>
#include <map>
#include <vector>

namespace vistk
{

/// A function which returns a \ref pipeline.
typedef boost::function<pipeline_t (config_t const& config)> pipeline_ctor_t;

/**
 * \class pipeline_registry
 *
 * \brief A registry of pipelines which can generate pipelines of a different types.
 *
 * \ingroup registries
 */
class VISTK_PIPELINE_EXPORT pipeline_registry
{
  public:
    /// The type of registry keys.
    typedef std::string type_t;
    /// The type for a description of the pipeline.
    typedef std::string description_t;
    /// A group of types.
    typedef std::vector<type_t> types_t;

    /**
     * \brief Destructor.
     */
    ~pipeline_registry();

    /**
     * \brief Adds a pipeline type to the registry.
     *
     * \throws pipeline_type_already_exists Thrown if the type already exists.
     *
     * \param type The name of the \ref pipeline type.
     * \param ctor The function which creates the pipeline of the \p type.
     */
    void register_pipeline(type_t const& type, description_t const& desc, pipeline_ctor_t ctor);
    /**
     * \brief Creates a pipeline of a specific type.
     *
     * \throws no_such_pipeline_type Thrown if the type is not known.
     *
     * \param type The name of the type of \ref pipeline to create.
     * \param config The configuration to pass the \ref pipeline.
     */
    pipeline_t create_pipeline(type_t const& type, config_t const& config) const;

    /**
     * \brief Returns all of the known types.
     */
    types_t types() const;

    /**
     * \brief Returns a pointer to the registry to use.
     */
    static pipeline_registry_t self();

    /// The default edge implementation to use.
    static type_t const default_type;
  private:
    pipeline_registry();

    static pipeline_registry_t m_self;

    typedef boost::tuple<description_t, pipeline_ctor_t> pipeline_typeinfo_t;
    typedef std::map<type_t, pipeline_typeinfo_t> pipeline_store_t;
    pipeline_store_t m_registry;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_PIPELINE_REGISTRY_H
