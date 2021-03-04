// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for detected_object_set_output process
 */

#ifndef _KWIVER_DETECTED_OBJECT_OUTPUT_PROCESS_H
#define _KWIVER_DETECTED_OBJECT_OUTPUT_PROCESS_H
#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
class KWIVER_PROCESSES_NO_EXPORT detected_object_output_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "detected_object_output",
               "Writes detected object sets to an output file.\n\n"
               "All detections are written to the same file." )

    detected_object_output_process( kwiver::vital::config_block_sptr const& config );
  virtual ~detected_object_output_process();

protected:
  void _configure() override;
  void _init() override;
  void _step() override;
  void _finalize() override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class detected_object_output_process

} // end namespace

#endif // _KWIVER_DETECTED_OBJECT_OUTPUT_PROCESS_H
