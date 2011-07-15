/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline_registry.h"
#include "pipeline_registry_exception.h"

namespace vistk
{

pipeline_registry_t pipeline_registry::m_self = pipeline_registry_t();
pipeline_registry::type_t const pipeline_registry::default_type = pipeline_registry::type_t("thread_per_process");

pipeline_registry
::~pipeline_registry()
{
}

void
pipeline_registry
::register_pipeline(type_t const& type, pipeline_ctor_t ctor)
{
  if (m_registry.find(type) != m_registry.end())
  {
    throw pipeline_type_already_exists(type);
  }

  m_registry[type] = ctor;
}

pipeline_t
pipeline_registry
::create_pipeline(type_t const& type, config_t const& config)
{
  if (m_registry.find(type) == m_registry.end())
  {
    throw no_such_pipeline_type(type);
  }

  return m_registry[type](config);
}

pipeline_registry::types_t
pipeline_registry
::types() const
{
  /// \todo Return all known types.
}

pipeline_registry_t
pipeline_registry
::self()
{
  if (!m_self)
  {
    m_self = pipeline_registry_t(new pipeline_registry);
  }

  return m_self;
}

pipeline_registry
::pipeline_registry()
{
}

} // end namespace vistk
