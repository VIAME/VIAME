/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_registry.h"
#include "process_registry_exception.h"

#include "config.h"
#include "process.h"
#include "types.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>

#include <map>
#include <set>
#include <utility>

/**
 * \file process_registry.cxx
 *
 * \brief Implementation of the \link sprokit::process_registry process registry\endlink.
 */

namespace sprokit
{

class process_registry::priv
{
  public:
    priv();
    ~priv();

    typedef boost::tuple<description_t, process_ctor_t> process_typeinfo_t;
    typedef std::map<process::type_t, process_typeinfo_t> process_store_t;
    process_store_t registry;

    typedef std::set<module_t> loaded_modules_t;
    loaded_modules_t loaded_modules;
};

static process_registry_t reg_self = process_registry_t();

process_registry
::~process_registry()
{
}

void
process_registry
::register_process(process::type_t const& type, description_t const& desc, process_ctor_t ctor)
{
  if (!ctor)
  {
    throw null_process_ctor_exception(type);
  }

  if (d->registry.count(type))
  {
    throw process_type_already_exists_exception(type);
  }

  d->registry[type] = priv::process_typeinfo_t(desc, ctor);
}

process_t
process_registry
::create_process(process::type_t const& type, process::name_t const& name, config_t const& config) const
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
  config->set_value(process::config_name, config::value_t(name));

  return i->second.get<1>()(config);
}

process::types_t
process_registry
::types() const
{
  process::types_t ts;

  BOOST_FOREACH (priv::process_store_t::value_type const& entry, d->registry)
  {
    process::type_t const& type = entry.first;

    ts.push_back(type);
  }

  return ts;
}

process_registry::description_t
process_registry
::description(process::type_t const& type) const
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
  return (0 != d->loaded_modules.count(module));
}

process_registry_t
process_registry
::self()
{
  typedef boost::shared_mutex mutex_t;

  static mutex_t mut;
  boost::upgrade_lock<mutex_t> lock(mut);

  if (!reg_self)
  {
    boost::upgrade_to_unique_lock<mutex_t> const write_lock(lock);

    (void)write_lock;

    if (!reg_self)
    {
      reg_self = process_registry_t(new process_registry);
    }
  }

  return reg_self;
}

process_registry
::process_registry()
  : d(new priv)
{
}

process_registry::priv
::priv()
  : registry()
  , loaded_modules()
{
}

process_registry::priv
::~priv()
{
}

}
