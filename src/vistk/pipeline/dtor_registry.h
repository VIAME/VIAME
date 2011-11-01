/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_DTOR_REGISTRY_H
#define VISTK_PIPELINE_DTOR_REGISTRY_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/function.hpp>
#include <boost/scoped_ptr.hpp>

#include <string>
#include <vector>

/**
 * \file dtor_registry.h
 *
 * \brief Header for the \link vistk::dtor_registry dtor registry\endlink.
 */

namespace vistk
{

/// A function cleans up at the program's end.
typedef boost::function<void ()> dtor_t;

/**
 * \class dtor_registry dtor_registry.h <vistk/pipeline/dtor_registry.h>
 *
 * \brief A registry of functions which must be called when the program ends.
 *
 * \ingroup registries
 */
class VISTK_PIPELINE_EXPORT dtor_registry
{
  public:
    /// The type of a module name.
    typedef std::string module_t;

    /**
     * \brief Destructor.
     */
    ~dtor_registry();

    /**
     * \brief Adds a dtor to the registry.
     *
     * \throws null_dtor_exception Thrown if \p dtor is \c NULL.
     *
     * \param dtor The function which must be called at program's end.
     */
    void register_dtor(dtor_t dtor);

    /**
     * \brief Marks a module as loaded.
     *
     * \param module The module to mark as loaded.
     */
    void mark_module_as_loaded(module_t const& module);
    /**
     * \brief Queries if a module has already been loaded.
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
    static dtor_registry_t self();
  private:
    dtor_registry();

    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PIPELINE_DTOR_REGISTRY_H
