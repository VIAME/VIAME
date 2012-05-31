/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_REGISTRY_H
#define VISTK_PIPELINE_PROCESS_REGISTRY_H

#include "pipeline-config.h"

#include "config.h"
#include "process.h"
#include "types.h"

#include <boost/function.hpp>
#include <boost/make_shared.hpp>
#include <boost/scoped_ptr.hpp>

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
    /// The type for a description of the pipeline.
    typedef std::string description_t;
    /// The type of a module name.
    typedef std::string module_t;

    /**
     * \brief Destructor.
     */
    ~process_registry();

    /**
     * \brief Add a process type to the registry.
     *
     * \throws null_process_ctor_exception Thrown if \p ctor is \c NULL.
     * \throws process_type_already_exists_exception Thrown if the type already exists.
     *
     * \see vistk::create_process
     *
     * \param type The name of the \ref process type.
     * \param desc A description of the type.
     * \param ctor The function which creates the process of the \p type.
     */
    void register_process(process::type_t const& type, description_t const& desc, process_ctor_t ctor);
    /**
     * \brief Create process of a specific type.
     *
     * \throws no_such_process_type_exception Thrown if the type is not known.
     *
     * \param type The type of \ref process to create.
     * \param config The configuration to pass the \ref process.
     *
     * \returns A new process of type \p type.
     */
    process_t create_process(process::type_t const& type, config_t const& config = config::empty_config()) const;

    /**
     * \brief Query for all available types.
     *
     * \returns All available types in the registry.
     */
    process::types_t types() const;
    /**
     * \brief Query for a description of a type.
     *
     * \param type The name of the type to description.
     *
     * \returns The description for the type \p type.
     */
    description_t description(process::type_t const& type) const;

    /**
     * \brief Mark a module as loaded.
     *
     * \param module The module to mark as loaded.
     */
    void mark_module_as_loaded(module_t const& module);
    /**
     * \brief Query if a module has already been loaded.
     *
     * \param module The module to query.
     *
     * \returns True if the module has already been loaded, false otherwise.
     */
    bool is_module_loaded(module_t const& module) const;

    /**
     * \brief Accessor to the registry.
     *
     * \returns The instance of the registry to use.
     */
    static process_registry_t self();
  private:
    process_registry();

    class priv;
    boost::scoped_ptr<priv> d;
};

/**
 * \brief A template function to create a process.
 *
 * This is to help reduce the amount of code needed in registration functions.
 *
 * \param conf The configuration to pass to the \ref process.
 *
 * \return The new process.
 */
template <typename T>
process_t
create_process(config_t const& conf)
{
  return boost::make_shared<T>(conf);
}

}

#endif // VISTK_PIPELINE_PROCESS_REGISTRY_H
