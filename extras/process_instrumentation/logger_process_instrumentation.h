// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for process instrumentation.
 */

#ifndef SPROKIT_LOGGER_PROCESS_INSTRUMENTATION_H
#define SPROKIT_LOGGER_PROCESS_INSTRUMENTATION_H

#include "instrumentation_plugin_export.h"
#include <sprokit/pipeline/process_instrumentation.h>
#include <vital/logger/logger.h>

namespace sprokit {

// -----------------------------------------------------------------
/**
 * \brief Process instrumentation using the logger.
 *
 * This class provides an implementation of process instrumentation
 * where each event is recorded to the logger.
 */
class INSTRUMENTATION_PLUGIN_NO_EXPORT logger_process_instrumentation
  : public process_instrumentation
{
public:
  logger_process_instrumentation();
  virtual ~logger_process_instrumentation() = default;

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
  void log_message( const std::string& data );

  kwiver::vital::logger_handle_t m_logger;
  kwiver::vital::kwiver_logger::log_level_t m_log_level;

}; // end class logger_process_instrumentation

} // end namespace

#endif // SPROKIT_LOGGER_PROCESS_INSTRUMENTATION_H
