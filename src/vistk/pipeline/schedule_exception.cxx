/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "schedule_exception.h"

#include <sstream>

/**
 * \file schedule_exception.cxx
 *
 * \brief Implementation of exceptions used within \link vistk::schedule schedules\endlink.
 */

namespace vistk
{

null_schedule_config_exception
::null_schedule_config_exception() throw()
  : schedule_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a schedule.";

  m_what = sstr.str();
}

null_schedule_config_exception
::~null_schedule_config_exception() throw()
{
}

char const*
null_schedule_config_exception
::what() const throw()
{
  return m_what.c_str();
}

}
