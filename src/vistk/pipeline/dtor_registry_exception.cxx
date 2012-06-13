/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "dtor_registry_exception.h"

#include <sstream>

/**
 * \file dtor_registry_exception.cxx
 *
 * \brief Implementation of exceptions used within the \link vistk::dtor_registry dtor registry\endlink.
 */

namespace vistk
{

null_dtor_exception
::null_dtor_exception() throw()
  : dtor_registry_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL dtor was passed to the registry";

  m_what = sstr.str();
}

null_dtor_exception
::~null_dtor_exception() throw()
{
}

}
