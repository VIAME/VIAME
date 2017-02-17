/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "logger_process_instrumentation.h"

#include <sprokit/pipeline/process.h>

namespace sprokit {

// ------------------------------------------------------------------
logger_process_instrumentation::
logger_process_instrumentation()
  : m_logger( kwiver::vital::get_logger( "sprokit.process_instrumentation" ) )
{ }


logger_process_instrumentation::
~logger_process_instrumentation()
{ }


// ------------------------------------------------------------------
void
logger_process_instrumentation::
  start_init_processing( std::string const& data )
{
  LOG_INFO( m_logger, process().name() << ": start_init_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_init_processing()
{
  LOG_INFO( m_logger, process().name() << ": stop_init_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_reset_processing( std::string const& data )
{
  LOG_INFO( m_logger, process().name() << ": stop_init_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_reset_processing()
{
  LOG_INFO( m_logger, process().name() << ": stop_reset_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_flush_processing( std::string const& data )
{
  LOG_INFO( m_logger, process().name() << ": start_reset_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_flush_processing()
{
  LOG_INFO( m_logger, process().name() << ": stop_flush_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_step_processing( std::string const& data )
{
  LOG_INFO( m_logger, process().name() << ": start_step_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_step_processing()
{
  LOG_INFO( m_logger, process().name() << ": stop_step_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_configure_processing( std::string const& data )
{
  LOG_INFO( m_logger, process().name() << ": start_configure_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_configure_processing()
{
  LOG_INFO( m_logger, process().name() << ": stop_configure_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_reconfigure_processing( std::string const& data )
{
  LOG_INFO( m_logger, process().name() << ": start_reconfigure_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_reconfigure_processing()
{
  LOG_INFO( m_logger, process().name() << ": stop_reconfigure_processing" );
}


} // end namespace
