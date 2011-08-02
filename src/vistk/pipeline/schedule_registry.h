/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_SCHEDULE_REGISTRY_H
#define VISTK_PIPELINE_SCHEDULE_REGISTRY_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/function.hpp>
#include <boost/tuple/tuple.hpp>

#include <map>
#include <string>
#include <vector>

/**
 * \file schedule_registry.h
 *
 * \brief Header for the \link vistk::schedule_registry schedule registry\endlink.
 */

namespace vistk
{

/// A function which returns a \ref schedule.
typedef boost::function<schedule_t (config_t const& config, pipeline_t const& pipe)> schedule_ctor_t;

/**
 * \class schedule_registry schedule_registry.h <vistk/pipeline/schedule_registry.h>
 *
 * \brief A registry of schedules which can generate schedules of a different types.
 *
 * \ingroup registries
 */
class VISTK_PIPELINE_EXPORT schedule_registry
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
    ~schedule_registry();

    /**
     * \brief Adds a schedule type to the registry.
     *
     * \throws schedule_type_already_exists_exception Thrown if the type already exists.
     *
     * \param type The name of the \ref schedule type.
     * \param desc A description of the type.
     * \param ctor The function which creates the schedule of the \p type.
     */
    void register_schedule(type_t const& type, description_t const& desc, schedule_ctor_t ctor);
    /**
     * \brief Creates schedule of a specific type.
     *
     * \throws no_such_schedule_type_exception Thrown if the type is not known.
     *
     * \param type The name of the type of \ref schedule to create.
     * \param config The configuration to pass the \ref schedule.
     * \param pipe The \ref pipeline to pass the \ref schedule.
     *
     * \returns A new schedule of type \p type.
     */
    schedule_t create_schedule(type_t const& type, config_t const& config, pipeline_t const& pipe) const;

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
    static schedule_registry_t self();

    /// The default schedule type.
    static type_t const default_type;
  private:
    schedule_registry();

    static schedule_registry_t m_self;

    typedef boost::tuple<description_t, schedule_ctor_t> schedule_typeinfo_t;
    typedef std::map<type_t, schedule_typeinfo_t> schedule_store_t;
    schedule_store_t m_registry;
};

}

#endif // VISTK_PIPELINE_SCHEDULE_REGISTRY_H
