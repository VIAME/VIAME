/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "scheduler_exception.h"

#include <sstream>

/**
 * \file scheduler_exception.cxx
 *
 * \brief Implementation of exceptions used within \link vistk::scheduler schedulers\endlink.
 */

namespace vistk
{

incompatible_pipeline_exception
::incompatible_pipeline_exception(std::string const& reason) throw()
  : scheduler_exception()
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

null_scheduler_config_exception
::null_scheduler_config_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL configuration was passed to a scheduler";

  m_what = sstr.str();
}

null_scheduler_config_exception
::~null_scheduler_config_exception() throw()
{
}

null_scheduler_pipeline_exception
::null_scheduler_pipeline_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "A NULL pipeline was passed to a scheduler";

  m_what = sstr.str();
}

null_scheduler_pipeline_exception
::~null_scheduler_pipeline_exception() throw()
{
}

}
