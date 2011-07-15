/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "edge_registry.h"
#include "edge_registry_exception.h"

#include <boost/foreach.hpp>

namespace vistk
{

edge_registry_t edge_registry::m_self = edge_registry_t();
edge_registry::type_t const edge_registry::default_type = edge_registry::type_t("dumb_pipe");

edge_registry
::~edge_registry()
{
}

void
edge_registry
::register_edge(type_t const& type, description_t const& desc, edge_ctor_t ctor)
{
  if (m_registry.find(type) != m_registry.end())
  {
    throw edge_type_already_exists(type);
  }

  m_registry[type] = edge_typeinfo_t(desc, ctor);
}

edge_t
edge_registry
::create_edge(type_t const& type, config_t const& config) const
{
  edge_store_t::const_iterator const i = m_registry.find(type);

  if (i == m_registry.end())
  {
    throw no_such_edge_type(type);
  }

  return i->second.get<1>()(config);
}

edge_registry::types_t
edge_registry
::types() const
{
  types_t types;

  BOOST_FOREACH (edge_store_t::value_type const& entry, m_registry)
  {
    types.push_back(entry.first);
  }

  return types;
}

edge_registry_t
edge_registry
::self()
{
  if (!m_self)
  {
    m_self = edge_registry_t(new edge_registry);
  }

  return m_self;
}

edge_registry
::edge_registry()
{
}

} // end namespace vistk
