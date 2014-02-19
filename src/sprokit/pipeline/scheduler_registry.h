/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPROKIT_PIPELINE_SCHEDULER_REGISTRY_H
#define SPROKIT_PIPELINE_SCHEDULER_REGISTRY_H

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
 * \brief Header for the \link sprokit::scheduler_registry scheduler registry\endlink.
 */

namespace sprokit
{

/// A function which returns a \ref scheduler.
typedef boost::function<scheduler_t (pipeline_t const& pipe, config_t const& config)> scheduler_ctor_t;

/**
 * \class scheduler_registry scheduler_registry.h <sprokit/pipeline/scheduler_registry.h>
 *
 * \brief A registry of schedulers which can generate schedulers of a different types.
 *
 * \ingroup registries
 */
class SPROKIT_PIPELINE_EXPORT scheduler_registry
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
     * \see sprokit::create_scheduler
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

    class SPROKIT_PIPELINE_NO_EXPORT priv;
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

#endif // SPROKIT_PIPELINE_SCHEDULER_REGISTRY_H
