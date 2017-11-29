/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

/**
 * \file
 * \brief Interface for process instrumentation.
 */

#ifndef SPROKIT_TIMING_PROCESS_INSTRUMENTATION_H
#define SPROKIT_TIMING_PROCESS_INSTRUMENTATION_H

#include "instrumentation_plugin_export.h"

#include <sprokit/pipeline/process_instrumentation.h>
#include <vital/logger/logger.h>
#include <vital/util/wall_timer.h>
#include <vital/util/cpu_timer.h>

#include <memory>

namespace sprokit {

// -----------------------------------------------------------------
/**
 * \brief Process instrumentation using the logger.
 *
 * This class provides an implementation of process instrumentation
 * where each event is recorded to the logger.
 *
 ** Note: The current implementation does not handle reentrant processes.
 */
class INSTRUMENTATION_PLUGIN_NO_EXPORT timing_process_instrumentation
  : public process_instrumentation
{
public:
  timing_process_instrumentation();
  virtual ~timing_process_instrumentation();

  virtual void start_init_processing( std::string const& data );
  virtual void stop_init_processing();

  virtual void start_reset_processing( std::string const& data );
  virtual void stop_reset_processing();

  virtual void start_flush_processing( std::string const& data );
  virtual void stop_flush_processing();

  virtual void start_step_processing( std::string const& data );
  virtual void stop_step_processing();

  virtual void start_configure_processing( std::string const& data );
  virtual void stop_configure_processing();

  virtual void start_reconfigure_processing( std::string const& data );
  virtual void stop_reconfigure_processing();

  virtual void configure( kwiver::vital::config_block_sptr const conf );
  virtual kwiver::vital::config_block_sptr get_configuration() const;

private:
  void write_interval( const std::string& tag, double interval );

  // The configured timer is allocated and stored here.
  std::shared_ptr<kwiver::vital::timer> m_timer;

  // The output file
  std::ofstream* m_output_file;

  kwiver::vital::logger_handle_t m_logger;

}; // end class timing_process_instrumentation

} // end namespace

#endif // SPROKIT_TIMING_PROCESS_INSTRUMENTATION_H
