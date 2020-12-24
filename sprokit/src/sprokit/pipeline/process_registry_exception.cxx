// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "process_registry_exception.h"

#include <sstream>

/**
 * \file process_registry_exception.cxx
 *
 * \brief Implementation of exceptions used within the \link sprokit::process_registry process registry\endlink.
 */

namespace sprokit
{

process_registry_exception
::process_registry_exception() noexcept
  : pipeline_exception()
{
}

process_registry_exception
::~process_registry_exception() noexcept
{
}

null_process_ctor_exception
::null_process_ctor_exception(process::type_t const& type) noexcept
  : process_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "A NULL constructor was passed for the "
          "process type \'" << m_type << "\'";

  m_what = sstr.str();
}

null_process_ctor_exception
::~null_process_ctor_exception() noexcept
{
}

null_process_registry_config_exception
::null_process_registry_config_exception() noexcept
  : process_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a process";

  m_what = sstr.str();
}

null_process_registry_config_exception
::~null_process_registry_config_exception() noexcept
{
}

no_such_process_type_exception
::no_such_process_type_exception(process::type_t const& type) noexcept
  : process_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is no such process of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

no_such_process_type_exception
::~no_such_process_type_exception() noexcept
{
}

process_type_already_exists_exception
::process_type_already_exists_exception(process::type_t const& type) noexcept
  : process_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is already a process of type \'" << type << "\' "
          "in the registry";

  m_what = sstr.str();
}

process_type_already_exists_exception
::~process_type_already_exists_exception() noexcept
{
}

}
