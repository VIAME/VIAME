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

#include <boost/foreach.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#include <utility>

/**
 * \file process_registry.cxx
 *
 * \brief Implementation of the \link vistk::process_registry process registry\endlink.
 */

namespace vistk
{

process_registry_t process_registry::m_self = process_registry_t();

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

  if (m_registry.find(type) != m_registry.end())
  {
    throw process_type_already_exists_exception(type);
  }

  m_registry[type] = process_typeinfo_t(desc, ctor);
}

process_t
process_registry
::create_process(type_t const& type, config_t const& config) const
{
  if (!config)
  {
    throw null_process_registry_config_exception();
  }

  process_store_t::const_iterator const i = m_registry.find(type);

  if (i == m_registry.end())
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

  BOOST_FOREACH (process_store_t::value_type const& entry, m_registry)
  {
    ts.push_back(entry.first);
  }

  return ts;
}

process_registry::description_t
process_registry
::description(type_t const& type) const
{
  process_store_t::const_iterator const i = m_registry.find(type);

  if (i == m_registry.end())
  {
    throw no_such_process_type_exception(type);
  }

  return i->second.get<0>();
}

process_registry_t
process_registry
::self()
{
  static boost::mutex mut;

  if (m_self)
  {
    return m_self;
  }

  boost::unique_lock<boost::mutex> lock(mut);
  if (!m_self)
  {
    m_self = process_registry_t(new process_registry);
  }

  return m_self;
}

process_registry
::process_registry()
{
}

}
