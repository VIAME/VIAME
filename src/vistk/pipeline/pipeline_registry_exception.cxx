/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipeline_registry_exception.h"

#include <sstream>

/**
 * \file pipeline_registry_exception.cxx
 *
 * \brief Implementation of exceptions used within the \link pipeline_registry pipeline registry\endlink.
 */

namespace vistk
{

no_such_pipeline_type
::no_such_pipeline_type(pipeline_registry::type_t const& type) throw()
  : pipeline_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is no such pipeline of type \'" << type << "\' "
       << "in the registry.";

  m_what = sstr.str();
}

no_such_pipeline_type
::~no_such_pipeline_type() throw()
{
}

char const*
no_such_pipeline_type
::what() const throw()
{
  return m_what.c_str();
}

pipeline_type_already_exists
::pipeline_type_already_exists(pipeline_registry::type_t const& type) throw()
  : pipeline_registry_exception()
  , m_type(type)
{
  std::ostringstream sstr;

  sstr << "There is already a pipeline of type \'" << type << "\' "
       << "in the registry.";

  m_what = sstr.str();
}

pipeline_type_already_exists
::~pipeline_type_already_exists() throw()
{
}

char const*
pipeline_type_already_exists
::what() const throw()
{
  return m_what.c_str();
}

} // end namespace vistk
