// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_EXTRAS_RT_PROCESS_INSTRUMENTATION_H
#define KWIVER_EXTRAS_RT_PROCESS_INSTRUMENTATION_H

#include "righttrack_plugin_export.h"
#include <sprokit/pipeline/process_instrumentation.h>

#include <RightTrack/BoundedEvent.h>

#include <memory>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Process instrumentation using RightTrack tool.
 *
 */
class RIGHTTRACK_PLUGIN_NO_EXPORT rt_process_instrumentation
: public process_instrumentation
{
public:
  // -- CONSTRUCTORS --
  rt_process_instrumentation();
  virtual ~rt_process_instrumentation() = default;

  void configure( kwiver::vital::config_block_sptr const config ) override;
  kwiver::vital::config_block_sptr get_configuration() const override;

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

private:
  std::unique_ptr< RightTrack::BoundedEvent > m_init_event;
  std::unique_ptr< RightTrack::BoundedEvent > m_finalize_event;
  std::unique_ptr< RightTrack::BoundedEvent > m_reset_event;
  std::unique_ptr< RightTrack::BoundedEvent > m_flush_event;
  std::unique_ptr< RightTrack::BoundedEvent > m_step_event;
  std::unique_ptr< RightTrack::BoundedEvent > m_configure_event;
  std::unique_ptr< RightTrack::BoundedEvent > m_reconfigure_event;
}; // end class rt_process_instrumentation

} // end namespace

#endif // KWIVER_EXTRAS_RT_PROCESS_INSTRUMENTATION_H
