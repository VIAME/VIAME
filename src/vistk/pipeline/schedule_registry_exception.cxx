/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "schedule_registry_exception.h"

#include <sstream>

/**
 * \file schedule_registry_exception.cxx
 *
 * \brief Implementation of exceptions used within the \link vistk::schedule_registry schedule registry\endlink.
 */

namespace vistk
{

null_schedule_ctor_exception
::null_schedule_ctor_exception(schedule_registry::type_t const& type) throw()
  : schedule_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "A NULL constructor was passed for the "
          "schedule type \'" << m_type << "\'";

  m_what = sstr.str();
}

null_schedule_ctor_exception
::~null_schedule_ctor_exception() throw()
{
}

null_schedule_registry_config_exception
::null_schedule_registry_config_exception() throw()
  : schedule_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a schedule";

  m_what = sstr.str();
}

null_schedule_registry_config_exception
::~null_schedule_registry_config_exception() throw()
{
}

null_schedule_registry_pipeline_exception
::null_schedule_registry_pipeline_exception() throw()
  : schedule_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL pipeline was passed to a schedule";

  m_what = sstr.str();
}

null_schedule_registry_pipeline_exception
::~null_schedule_registry_pipeline_exception() throw()
{
}

no_such_schedule_type_exception
::no_such_schedule_type_exception(schedule_registry::type_t const& type) throw()
  : schedule_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is no such schedule of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

no_such_schedule_type_exception
::~no_such_schedule_type_exception() throw()
{
}

schedule_type_already_exists_exception
::schedule_type_already_exists_exception(schedule_registry::type_t const& type) throw()
  : schedule_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is already a schedule of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

schedule_type_already_exists_exception
::~schedule_type_already_exists_exception() throw()
{
}

}
