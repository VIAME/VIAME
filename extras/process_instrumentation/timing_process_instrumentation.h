// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
#include <vital/util/simple_stats.h>

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
 *
 * Timing information can be obtained on a per-process basis by using
 * this process instrumentation provider. The provider is configured
 * in the pipe file as shown below:
 *
\code
process motion :: detect_motion
    block _instrumentation
       type = timing
    endblock
\endcode
 */
class INSTRUMENTATION_PLUGIN_NO_EXPORT timing_process_instrumentation
  : public process_instrumentation
{
public:
  timing_process_instrumentation();
  virtual ~timing_process_instrumentation();

  void start_init_processing( std::string const& data ) override;
  void stop_init_processing() override;

  void start_finalize_processing( std::string const& data ) override;
  void stop_finalize_processing() override;

  void start_reset_processing( std::string const& data ) override;
  void stop_reset_processing() override;

  void start_flush_processing( std::string const& data ) override;
  void stop_flush_processing() override;

  void start_step_processing( std::string const& data ) override;
  void stop_step_processing() override;

  void start_configure_processing( std::string const& data ) override;
  void stop_configure_processing() override;

  void start_reconfigure_processing( std::string const& data ) override;
  void stop_reconfigure_processing() override;

  void configure( kwiver::vital::config_block_sptr const conf ) override;
  kwiver::vital::config_block_sptr get_configuration() const override;

private:
  void write_interval( const std::string& tag, double interval );

  // The configured timer is allocated and stored here.
  std::shared_ptr<kwiver::vital::timer> m_timer;

  // The output file
  std::ofstream* m_output_file;

  kwiver::vital::logger_handle_t m_logger;

  // Statistics for step, since it is called frequently.
  kwiver::vital::simple_stats m_step_stats;

}; // end class timing_process_instrumentation

} // end namespace

#endif // SPROKIT_TIMING_PROCESS_INSTRUMENTATION_H
