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

/**
 * \file process_registry.h
 *
 * \brief Header for the \link vistk::process_registry process registry\endlink.
 */

namespace vistk
{

/// A function which returns a \ref process.
typedef boost::function<process_t (config_t const& config)> process_ctor_t;

/**
 * \class process_registry process_registry.h <vistk/pipeline/process_registry.h>
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
     * \param desc A description of the type.
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
     *
     * \returns A new process of type \p type.
     */
    process_t create_process(type_t const& type, config_t const& config) const;

    /**
     * \brief Query for all available types.
     *
     * \returns All available types in the registry.
     */
    types_t types() const;
    /**
     * \brief Query for a description of a type.
     *
     * \param type The name of the type to description.
     *
     * \returns The description for the type \p type.
     */
    description_t description(type_t const& type) const;

    /**
     * \brief Accessor to the registry.
     *
     * \returns The instance of the registry to use.
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
