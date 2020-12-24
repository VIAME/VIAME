// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "scheduler_registry_exception.h"

#include <sstream>

/**
 * \file scheduler_registry_exception.cxx
 *
 * \brief Implementation of exceptions used within the \link sprokit::scheduler_registry scheduler registry\endlink.
 */

namespace sprokit{

// ------------------------------------------------------------------
scheduler_registry_exception
::scheduler_registry_exception() noexcept
  : pipeline_exception()
{
}

// ------------------------------------------------------------------
scheduler_registry_exception
::~scheduler_registry_exception() noexcept
{
}

// ------------------------------------------------------------------
null_scheduler_ctor_exception
::null_scheduler_ctor_exception(sprokit::scheduler::type_t const& type) noexcept
  : scheduler_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "A NULL constructor was passed for the "
          "scheduler type \'" << m_type << "\'";

  m_what = sstr.str();
}

// ------------------------------------------------------------------
null_scheduler_ctor_exception
::~null_scheduler_ctor_exception() noexcept
{
}

// ------------------------------------------------------------------
null_scheduler_registry_config_exception
::null_scheduler_registry_config_exception() noexcept
  : scheduler_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a scheduler";

  m_what = sstr.str();
}

// ------------------------------------------------------------------
null_scheduler_registry_config_exception
::~null_scheduler_registry_config_exception() noexcept
{
}

// ------------------------------------------------------------------
null_scheduler_registry_pipeline_exception
::null_scheduler_registry_pipeline_exception() noexcept
  : scheduler_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL pipeline was passed to a scheduler";

  m_what = sstr.str();
}

// ------------------------------------------------------------------
null_scheduler_registry_pipeline_exception
::~null_scheduler_registry_pipeline_exception() noexcept
{
}

// ------------------------------------------------------------------
no_such_scheduler_type_exception
::no_such_scheduler_type_exception(sprokit::scheduler::type_t const& type) noexcept
  : scheduler_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is no such scheduler of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

// ------------------------------------------------------------------
no_such_scheduler_type_exception
::~no_such_scheduler_type_exception() noexcept
{
}

// ------------------------------------------------------------------
scheduler_type_already_exists_exception
::scheduler_type_already_exists_exception(sprokit::scheduler::type_t const& type) noexcept
  : scheduler_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is already a scheduler of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

// ------------------------------------------------------------------
scheduler_type_already_exists_exception
::~scheduler_type_already_exists_exception() noexcept
{
}

} // end namespace
