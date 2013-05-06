/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "scheduler_registry_exception.h"

#include <sstream>

/**
 * \file scheduler_registry_exception.cxx
 *
 * \brief Implementation of exceptions used within the \link sprokit::scheduler_registry scheduler registry\endlink.
 */

namespace sprokit
{

scheduler_registry_exception
::scheduler_registry_exception() throw()
  : pipeline_exception()
{
}

scheduler_registry_exception
::~scheduler_registry_exception() throw()
{
}

null_scheduler_ctor_exception
::null_scheduler_ctor_exception(scheduler_registry::type_t const& type) throw()
  : scheduler_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "A NULL constructor was passed for the "
          "scheduler type \'" << m_type << "\'";

  m_what = sstr.str();
}

null_scheduler_ctor_exception
::~null_scheduler_ctor_exception() throw()
{
}

null_scheduler_registry_config_exception
::null_scheduler_registry_config_exception() throw()
  : scheduler_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a scheduler";

  m_what = sstr.str();
}

null_scheduler_registry_config_exception
::~null_scheduler_registry_config_exception() throw()
{
}

null_scheduler_registry_pipeline_exception
::null_scheduler_registry_pipeline_exception() throw()
  : scheduler_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL pipeline was passed to a scheduler";

  m_what = sstr.str();
}

null_scheduler_registry_pipeline_exception
::~null_scheduler_registry_pipeline_exception() throw()
{
}

no_such_scheduler_type_exception
::no_such_scheduler_type_exception(scheduler_registry::type_t const& type) throw()
  : scheduler_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is no such scheduler of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

no_such_scheduler_type_exception
::~no_such_scheduler_type_exception() throw()
{
}

scheduler_type_already_exists_exception
::scheduler_type_already_exists_exception(scheduler_registry::type_t const& type) throw()
  : scheduler_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is already a scheduler of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

scheduler_type_already_exists_exception
::~scheduler_type_already_exists_exception() throw()
{
}

}
