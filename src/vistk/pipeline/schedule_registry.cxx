/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "schedule_registry.h"
#include "schedule_registry_exception.h"

#include "types.h"

#include <boost/foreach.hpp>

#include <utility>

/**
 * \file schedule_registry.cxx
 *
 * \brief Implementation of the \link vistk::schedule_registry schedule registry\endlink.
 */

namespace vistk
{

schedule_registry_t schedule_registry::m_self = schedule_registry_t();
schedule_registry::type_t const schedule_registry::default_type = type_t("thread_pool");

schedule_registry
::~schedule_registry()
{
}

void
schedule_registry
::register_schedule(type_t const& type, description_t const& desc, schedule_ctor_t ctor)
{
  if (m_registry.find(type) != m_registry.end())
  {
    throw schedule_type_already_exists(type);
  }

  m_registry[type] = schedule_typeinfo_t(desc, ctor);
}

schedule_t
schedule_registry
::create_schedule(type_t const& type, config_t const& config, pipeline_t const& pipe) const
{
  schedule_store_t::const_iterator const i = m_registry.find(type);

  if (i == m_registry.end())
  {
    throw no_such_schedule_type(type);
  }

  return i->second.get<1>()(config, pipe);
}

schedule_registry::types_t
schedule_registry
::types() const
{
  types_t ts;

  BOOST_FOREACH (schedule_store_t::value_type const& entry, m_registry)
  {
    ts.push_back(entry.first);
  }

  return ts;
}

schedule_registry::description_t
schedule_registry
::description(type_t const& type) const
{
  schedule_store_t::const_iterator const i = m_registry.find(type);

  if (i == m_registry.end())
  {
    throw no_such_schedule_type(type);
  }

  return i->second.get<0>();
}

schedule_registry_t
schedule_registry
::self()
{
  if (!m_self)
  {
    m_self = schedule_registry_t(new schedule_registry);
  }

  return m_self;
}

schedule_registry
::schedule_registry()
{
}

} // end namespace vistk
