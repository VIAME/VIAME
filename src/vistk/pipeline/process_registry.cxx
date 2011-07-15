/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_registry.h"
#include "process_registry_exception.h"

#include "types.h"

#include <utility>

namespace vistk
{

process_registry_t process_registry::m_self = process_registry_t();

process_registry
::~process_registry()
{
}

void
process_registry
::register_process(type_t const& type, process_ctor_t ctor)
{
  if (m_registry.find(type) != m_registry.end())
  {
    throw process_type_already_exists(type);
  }

  m_registry[type] = ctor;
}

process_t
process_registry
::create_process(type_t const& type, config_t const& config)
{
  if (m_registry.find(type) == m_registry.end())
  {
    throw no_such_process_type(type);
  }

  return m_registry[type](config);
}

process_registry::types_t
process_registry
::types() const
{
  /// \todo Return all known types.
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
