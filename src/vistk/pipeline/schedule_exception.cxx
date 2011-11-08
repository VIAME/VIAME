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

incompatible_pipeline_exception
::incompatible_pipeline_exception(std::string const& reason) throw()
  : schedule_exception()
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "The pipeline cannot be run: " << m_reason;

  m_what = sstr.str();
}

incompatible_pipeline_exception
::~incompatible_pipeline_exception() throw()
{
}

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

null_schedule_pipeline_exception
::null_schedule_pipeline_exception() throw()
  : schedule_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL pipeline was passed to a schedule.";

  m_what = sstr.str();
}

null_schedule_pipeline_exception
::~null_schedule_pipeline_exception() throw()
{
}

}
