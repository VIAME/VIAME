/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "scheduler_registry.h"
#include "scheduler_registry_exception.h"

#include "types.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>

#include <map>
#include <set>

/**
 * \file scheduler_registry.cxx
 *
 * \brief Implementation of the \link vistk::scheduler_registry scheduler registry\endlink.
 */

namespace vistk
{

scheduler_registry::type_t const scheduler_registry::default_type = type_t("thread_per_process");

class scheduler_registry::priv
{
  public:
    priv();
    ~priv();

    typedef boost::tuple<description_t, scheduler_ctor_t> scheduler_typeinfo_t;
    typedef std::map<type_t, scheduler_typeinfo_t> scheduler_store_t;
    scheduler_store_t registry;

    typedef std::set<module_t> loaded_modules_t;
    loaded_modules_t loaded_modules;
};

static scheduler_registry_t reg_self = scheduler_registry_t();

scheduler_registry
::~scheduler_registry()
{
}

void
scheduler_registry
::register_scheduler(type_t const& type, description_t const& desc, scheduler_ctor_t ctor)
{
  if (!ctor)
  {
    throw null_scheduler_ctor_exception(type);
  }

  if (d->registry.find(type) != d->registry.end())
  {
    throw scheduler_type_already_exists_exception(type);
  }

  d->registry[type] = priv::scheduler_typeinfo_t(desc, ctor);
}

scheduler_t
scheduler_registry
::create_scheduler(type_t const& type, pipeline_t const& pipe, config_t const& config) const
{
  if (!config)
  {
    throw null_scheduler_registry_config_exception();
  }

  if (!pipe)
  {
    throw null_scheduler_registry_pipeline_exception();
  }

  priv::scheduler_store_t::const_iterator const i = d->registry.find(type);

  if (i == d->registry.end())
  {
    throw no_such_scheduler_type_exception(type);
  }

  return i->second.get<1>()(pipe, config);
}

scheduler_registry::types_t
scheduler_registry
::types() const
{
  types_t ts;

  BOOST_FOREACH (priv::scheduler_store_t::value_type const& entry, d->registry)
  {
    type_t const& type = entry.first;

    ts.push_back(type);
  }

  return ts;
}

scheduler_registry::description_t
scheduler_registry
::description(type_t const& type) const
{
  priv::scheduler_store_t::const_iterator const i = d->registry.find(type);

  if (i == d->registry.end())
  {
    throw no_such_scheduler_type_exception(type);
  }

  return i->second.get<0>();
}

void
scheduler_registry
::mark_module_as_loaded(module_t const& module)
{
  d->loaded_modules.insert(module);
}

bool
scheduler_registry
::is_module_loaded(module_t const& module) const
{
  priv::loaded_modules_t::const_iterator const i = d->loaded_modules.find(module);

  return (i != d->loaded_modules.end());
}

scheduler_registry_t
scheduler_registry
::self()
{
  static boost::mutex mut;

  if (reg_self)
  {
    return reg_self;
  }

  {
    boost::mutex::scoped_lock const lock(mut);

    (void)lock;

    if (!reg_self)
    {
      reg_self = scheduler_registry_t(new scheduler_registry);
    }
  }

  return reg_self;
}

scheduler_registry
::scheduler_registry()
  : d(new priv)
{
}

scheduler_registry::priv
::priv()
{
}

scheduler_registry::priv
::~priv()
{
}

}
