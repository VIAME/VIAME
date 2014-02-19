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

#ifndef SPROKIT_PIPELINE_PROCESS_REGISTRY_H
#define SPROKIT_PIPELINE_PROCESS_REGISTRY_H

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
 * \brief Header for the \link sprokit::process_registry process registry\endlink.
 */

namespace sprokit
{

/// A function which returns a \ref process.
typedef boost::function<process_t (config_t const& config)> process_ctor_t;

/**
 * \class process_registry process_registry.h <sprokit/pipeline/process_registry.h>
 *
 * \brief A registry of processes which can generate processes of a different types.
 *
 * \ingroup registries
 */
class SPROKIT_PIPELINE_EXPORT process_registry
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
     * \see sprokit::create_process
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
     * \param name The name of the \ref process to create.
     * \param config The configuration to pass the \ref process.
     *
     * \returns A new process of type \p type.
     */
    process_t create_process(process::type_t const& type, process::name_t const& name, config_t const& config = config::empty_config()) const;

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

    class SPROKIT_PIPELINE_NO_EXPORT priv;
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

#endif // SPROKIT_PIPELINE_PROCESS_REGISTRY_H
