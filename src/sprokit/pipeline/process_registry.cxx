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
  typedef boost::mutex mutex_t;

  static mutex_t mut;

  if (reg_self)
  {
    return reg_self;
  }

  {
    boost::unique_lock<mutex_t> const lock(mut);

    (void)lock;

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
