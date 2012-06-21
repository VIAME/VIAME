/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_SCHEDULER_REGISTRY_H
#define VISTK_PIPELINE_SCHEDULER_REGISTRY_H

#include "pipeline-config.h"

#include "config.h"
#include "types.h"

#include <boost/function.hpp>
#include <boost/make_shared.hpp>
#include <boost/scoped_ptr.hpp>

#include <string>
#include <vector>

/**
 * \file scheduler_registry.h
 *
 * \brief Header for the \link vistk::scheduler_registry scheduler registry\endlink.
 */

namespace vistk
{

/// A function which returns a \ref scheduler.
typedef boost::function<scheduler_t (pipeline_t const& pipe, config_t const& config)> scheduler_ctor_t;

/**
 * \class scheduler_registry scheduler_registry.h <vistk/pipeline/scheduler_registry.h>
 *
 * \brief A registry of schedulers which can generate schedulers of a different types.
 *
 * \ingroup registries
 */
class VISTK_PIPELINE_EXPORT scheduler_registry
{
  public:
    /// The type of registry keys.
    typedef std::string type_t;
    /// The type for a description of the pipeline.
    typedef std::string description_t;
    /// A group of types.
    typedef std::vector<type_t> types_t;
    /// The type of a module name.
    typedef std::string module_t;

    /**
     * \brief Destructor.
     */
    ~scheduler_registry();

    /**
     * \brief Add a scheduler type to the registry.
     *
     * \throws null_scheduler_ctor_exception Thrown if \p ctor is \c NULL.
     * \throws scheduler_type_already_exists_exception Thrown if the type already exists.
     *
     * \see vistk::create_scheduler
     *
     * \param type The name of the \ref scheduler type.
     * \param desc A description of the type.
     * \param ctor The function which creates the scheduler of the \p type.
     */
    void register_scheduler(type_t const& type, description_t const& desc, scheduler_ctor_t ctor);
    /**
     * \brief Create scheduler of a specific type.
     *
     * \throws no_such_scheduler_type_exception Thrown if the type is not known.
     *
     * \param type The name of the type of \ref scheduler to create.
     * \param pipe The \ref pipeline to pass the \ref scheduler.
     * \param config The configuration to pass the \ref scheduler.
     *
     * \returns A new scheduler of type \p type.
     */
    scheduler_t create_scheduler(type_t const& type, pipeline_t const& pipe, config_t const& config = config::empty_config()) const;

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
    static scheduler_registry_t self();

    /// The default scheduler type.
    static type_t const default_type;
  private:
    scheduler_registry();

    class priv;
    boost::scoped_ptr<priv> d;
};

/**
 * \brief A template function to create a scheduler.
 *
 * This is to help reduce the amount of code needed in registration functions.
 *
 * \param conf The configuration to pass to the \ref scheduler.
 * \param pipe The \ref pipeline to pass the \ref scheduler.
 *
 * \return The new scheduler.
 */
template <typename T>
scheduler_t
create_scheduler(pipeline_t const& pipe, config_t const& conf)
{
  return boost::make_shared<T>(pipe, conf);
}

}

#endif // VISTK_PIPELINE_SCHEDULER_REGISTRY_H
