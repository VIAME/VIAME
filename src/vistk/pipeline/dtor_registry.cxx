/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "dtor_registry.h"
#include "dtor_registry_exception.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>

#include <map>
#include <set>
#include <utility>

/**
 * \file dtor_registry.cxx
 *
 * \brief Implementation of the \link vistk::dtor_registry dtor registry\endlink.
 */

namespace vistk
{

class dtor_registry::priv
{
  public:
    priv();
    ~priv();

    typedef std::vector<dtor_t> dtor_store_t;
    dtor_store_t registry;

    typedef std::set<module_t> loaded_modules_t;
    loaded_modules_t loaded_modules;
};

static dtor_registry_t reg_self = dtor_registry_t();

dtor_registry
::~dtor_registry()
{
  BOOST_FOREACH (dtor_t dtor, d->registry)
  {
    dtor();
  }
}

void
dtor_registry
::register_dtor(dtor_t dtor)
{
  if (!dtor)
  {
    throw null_dtor_exception();
  }

  d->registry.push_back(dtor);
}

void
dtor_registry
::mark_module_as_loaded(module_t const& module)
{
  d->loaded_modules.insert(module);
}

bool
dtor_registry
::is_module_loaded(module_t const& module) const
{
  priv::loaded_modules_t::const_iterator const i = d->loaded_modules.find(module);

  return (i != d->loaded_modules.end());
}

dtor_registry_t
dtor_registry
::self()
{
  static boost::mutex mut;

  if (reg_self)
  {
    return reg_self;
  }

  boost::unique_lock<boost::mutex> lock(mut);
  if (!reg_self)
  {
    reg_self = dtor_registry_t(new dtor_registry);
  }

  return reg_self;
}

dtor_registry
::dtor_registry()
  : d(new priv)
{
}

dtor_registry::priv
::priv()
{
}

dtor_registry::priv
::~priv()
{
}

}
