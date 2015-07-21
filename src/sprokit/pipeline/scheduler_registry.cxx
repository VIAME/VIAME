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

#include "scheduler_registry.h"
#include "scheduler_registry_exception.h"

#include "types.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>

#include <map>
#include <set>

/**
 * \file scheduler_registry.cxx
 *
 * \brief Implementation of the \link sprokit::scheduler_registry scheduler registry\endlink.
 */

namespace sprokit
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

  if (d->registry.count(type))
  {
    throw scheduler_type_already_exists_exception(type);
  }

  d->registry[type] = priv::scheduler_typeinfo_t(desc, ctor);
}

scheduler_t
scheduler_registry
::create_scheduler(type_t const& type, pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config) const
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
  return (0 != d->loaded_modules.count(module));
}

scheduler_registry_t
scheduler_registry
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
  : registry()
  , loaded_modules()
{
}

scheduler_registry::priv
::~priv()
{
}

}
