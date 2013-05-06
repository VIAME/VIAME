/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "scheduler_exception.h"

#include <sstream>

/**
 * \file scheduler_exception.cxx
 *
 * \brief Implementation of exceptions used within \link sprokit::scheduler schedulers\endlink.
 */

namespace sprokit
{

scheduler_exception
::scheduler_exception() throw()
  : pipeline_exception()
{
}

scheduler_exception
::~scheduler_exception() throw()
{
}

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

restart_scheduler_exception
::restart_scheduler_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "A scheduler was restarted";

  m_what = sstr.str();
}

restart_scheduler_exception
::~restart_scheduler_exception() throw()
{
}

wait_before_start_exception
::wait_before_start_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "A scheduler was waited on before it was started";

  m_what = sstr.str();
}

wait_before_start_exception
::~wait_before_start_exception() throw()
{
}

pause_before_start_exception
::pause_before_start_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "A scheduler was paused before it was started";

  m_what = sstr.str();
}

pause_before_start_exception
::~pause_before_start_exception() throw()
{
}

repause_scheduler_exception
::repause_scheduler_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "A paused scheduler was paused";

  m_what = sstr.str();
}

repause_scheduler_exception
::~repause_scheduler_exception() throw()
{
}

resume_before_start_exception
::resume_before_start_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "A scheduler was resumed before it was started";

  m_what = sstr.str();
}

resume_before_start_exception
::~resume_before_start_exception() throw()
{
}

resume_unpaused_scheduler_exception
::resume_unpaused_scheduler_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "An unpaused scheduler was resumed";

  m_what = sstr.str();
}

resume_unpaused_scheduler_exception
::~resume_unpaused_scheduler_exception() throw()
{
}

stop_before_start_exception
::stop_before_start_exception() throw()
  : scheduler_exception()
{
  std::ostringstream sstr;

  sstr << "A scheduler was stopped before it was started";

  m_what = sstr.str();
}

stop_before_start_exception
::~stop_before_start_exception() throw()
{
}

}
