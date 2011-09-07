/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_registry.h"
#include "process_registry_exception.h"

#include "config.h"
#include "process.h"
#include "types.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>

#include <map>
#include <set>
#include <utility>

/**
 * \file process_registry.cxx
 *
 * \brief Implementation of the \link vistk::process_registry process registry\endlink.
 */

namespace vistk
{

class process_registry::priv
{
  public:
    priv();
    ~priv();

    static process_registry_t self;

    typedef boost::tuple<description_t, process_ctor_t> process_typeinfo_t;
    typedef std::map<type_t, process_typeinfo_t> process_store_t;
    process_store_t registry;

    typedef std::set<module_t> loaded_modules_t;
    loaded_modules_t loaded_modules;
};

process_registry_t process_registry::priv::self = process_registry_t();

process_registry
::~process_registry()
{
}

void
process_registry
::register_process(type_t const& type, description_t const& desc, process_ctor_t ctor)
{
  if (!ctor)
  {
    throw null_process_ctor_exception(type);
  }

  if (d->registry.find(type) != d->registry.end())
  {
    throw process_type_already_exists_exception(type);
  }

  d->registry[type] = priv::process_typeinfo_t(desc, ctor);
}

process_t
process_registry
::create_process(type_t const& type, config_t const& config) const
{
  if (!config)
  {
    throw null_process_registry_config_exception();
  }

  priv::process_store_t::const_iterator const i = d->registry.find(type);

  if (i == d->registry.end())
  {
    throw no_such_process_type_exception(type);
  }

  config->set_value(process::config_type, config::value_t(type));

  return i->second.get<1>()(config);
}

process_registry::types_t
process_registry
::types() const
{
  types_t ts;

  BOOST_FOREACH (priv::process_store_t::value_type const& entry, d->registry)
  {
    ts.push_back(entry.first);
  }

  return ts;
}

process_registry::description_t
process_registry
::description(type_t const& type) const
{
  priv::process_store_t::const_iterator const i = d->registry.find(type);

  if (i == d->registry.end())
  {
    throw no_such_process_type_exception(type);
  }

  return i->second.get<0>();
}

void
process_registry
::mark_module_as_loaded(module_t const& module)
{
  d->loaded_modules.insert(module);
}

bool
process_registry
::is_module_loaded(module_t const& module) const
{
  priv::loaded_modules_t::const_iterator const i = d->loaded_modules.find(module);

  return (i != d->loaded_modules.end());
}

process_registry_t
process_registry
::self()
{
  static boost::mutex mut;

  if (priv::self)
  {
    return priv::self;
  }

  boost::unique_lock<boost::mutex> lock(mut);
  if (!priv::self)
  {
    priv::self = process_registry_t(new process_registry);
  }

  return priv::self;
}

process_registry
::process_registry()
{
  d.reset(new priv);
}

process_registry::priv
::priv()
{
}

process_registry::priv
::~priv()
{
}

}
