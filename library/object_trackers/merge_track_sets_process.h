// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#pragma once

#include <viame/pipeline_framework/process.h>

#include "viame_processes_object_trackers_export.h"

#include <memory>

namespace kwiver {

// ----------------------------------------------------------------

/**
 * \class merge_track_sets_process
 *
 * \brief Merges two or more track sets
 *
 * \iports
 * \iport{image}
 *
 * \oports
 * \oport{image1}
 * \oport{image2}
 *
 */
class VIAME_PROCESSES_OBJECT_TRACKERS_EXPORT merge_track_sets_process
  : public sprokit::process
{
public:
  PLUGIN_INFO(
    "merge_track_sets",
    "Merge multiple input track sets into one output set." )

  merge_track_sets_process( kwiver::vital::config_block_sptr const& config );
  virtual ~merge_track_sets_process();

protected:
  virtual void _configure();
  virtual void _step();
  virtual sprokit::process::port_info_t _input_port_info( port_t const& port );

private:
  void make_ports();
  void make_config();

  class priv;

  const std::unique_ptr< priv > d;
}; // end class merge_track_sets_process

} // end namespace
