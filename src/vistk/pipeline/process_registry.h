/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_REGISTRY_H
#define VISTK_PIPELINE_PROCESS_REGISTRY_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/function.hpp>
#include <boost/tuple/tuple.hpp>

#include <map>
#include <string>
#include <vector>

namespace vistk
{

/// A function which returns a \ref process.
typedef boost::function<process_t (config_t const& config)> process_ctor_t;

/**
 * \class process_registry
 *
 * \brief A registry of processes which can generate processes of a different types.
 *
 * \ingroup registries
 */
class VISTK_PIPELINE_EXPORT process_registry
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
    ~process_registry();

    /**
     * \brief Adds a process type to the registry.
     *
     * \throws process_type_already_exists Thrown if the type already exists.
     *
     * \param type The name of the \ref process type.
     * \param ctor The function which creates the process of the \p type.
     */
    void register_process(type_t const& type, description_t const& desc, process_ctor_t ctor);
    /**
     * \brief Creates process of a specific type.
     *
     * \throws no_such_process_type Thrown if the type is not known.
     *
     * \param type The name of the type of \ref process to create.
     * \param config The configuration to pass the \ref process.
     */
    process_t create_process(type_t const& type, config_t const& config) const;

    /**
     * \brief Returns all of the known types.
     */
    types_t types() const;

    /**
     * \brief Returns a pointer to the registry to use.
     */
    static process_registry_t self();
  private:
    process_registry();

    static process_registry_t m_self;

    typedef boost::tuple<description_t, process_ctor_t> process_typeinfo_t;
    typedef std::map<type_t, process_typeinfo_t> process_store_t;
    process_store_t m_registry;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_PROCESS_REGISTRY_H
