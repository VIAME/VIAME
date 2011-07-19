/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "edge_registry_exception.h"

#include <sstream>

/**
 * \file edge_registry_exception.cxx
 *
 * \brief Implementation of exceptions used within the \link edge_registry edge registry\endlink.
 */

namespace vistk
{

no_such_edge_type
::no_such_edge_type(edge_registry::type_t const& type) throw()
  : edge_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is no such edge of type \'" << type << "\' "
       << "in the registry.";

  m_what = sstr.str();
}

no_such_edge_type
::~no_such_edge_type() throw()
{
}

char const*
no_such_edge_type
::what() const throw()
{
  return m_what.c_str();
}

edge_type_already_exists
::edge_type_already_exists(edge_registry::type_t const& type) throw()
  : edge_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is already a edge of type \'" << type << "\' "
       << "in the registry.";

  m_what = sstr.str();
}

edge_type_already_exists
::~edge_type_already_exists() throw()
{
}

char const*
edge_type_already_exists
::what() const throw()
{
  return m_what.c_str();
}

} // end namespace vistk
