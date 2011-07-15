/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_registry_exception.h"

#include <sstream>

namespace vistk
{

no_such_process_type
::no_such_process_type(process_registry::type_t const& type) throw()
  : process_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is no such process of type \'" << type << "\' "
       << "in the registry.";

  m_what = sstr.str();
}

no_such_process_type
::~no_such_process_type() throw()
{
}

char const*
no_such_process_type
::what() const throw()
{
  return m_what.c_str();
}

process_type_already_exists
::process_type_already_exists(process_registry::type_t const& type) throw()
  : process_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is already a process of type \'" << type << "\' "
       << "in the registry.";

  m_what = sstr.str();
}

process_type_already_exists
::~process_type_already_exists() throw()
{
}

char const*
process_type_already_exists
::what() const throw()
{
  return m_what.c_str();
}

} // end namespace vistk
