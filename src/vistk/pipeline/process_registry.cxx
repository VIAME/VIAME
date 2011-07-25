/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_registry.h"
#include "process_registry_exception.h"

#include "types.h"

#include <boost/foreach.hpp>

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
  if (m_registry.find(type) != m_registry.end())
  {
    throw process_type_already_exists(type);
  }

  m_registry[type] = process_typeinfo_t(desc, ctor);
}

process_t
process_registry
::create_process(type_t const& type, config_t const& config) const
{
  process_store_t::const_iterator const i = m_registry.find(type);

  if (i == m_registry.end())
  {
    throw no_such_process_type(type);
  }

  return i->second.get<1>()(config);
}

process_registry::types_t
process_registry
::types() const
{
  types_t types;

  BOOST_FOREACH (process_store_t::value_type const& entry, m_registry)
  {
    types.push_back(entry.first);
  }

  return types;
}

process_registry::description_t
process_registry
::description(type_t const& type) const
{
  process_store_t::const_iterator const i = m_registry.find(type);

  if (i == m_registry.end())
  {
    throw no_such_process_type(type);
  }

  return i->second.get<0>();
}

process_registry_t
process_registry
::self()
{
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

} // end namespace vistk
