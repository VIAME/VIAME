// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for track_descriptor_set_output process
 */

#ifndef _KWIVER_WRITE_TRACK_DESCRIPTOR_PROCESS_H
#define _KWIVER_WRITE_TRACK_DESCRIPTOR_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

  // -----------------------------------------------------------------------------
/**
 * \class write_track_descriptor_process
 *
 * \brief Write a set of track descriptors to a file
 *
 * \iports
 * \iport{track descriptor_set}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT write_track_descriptor_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "write_track_descriptor",
               "Writes track descriptor sets to an output file. "
               "All descriptors are written to the same file." )

  write_track_descriptor_process( kwiver::vital::config_block_sptr const& config );
  virtual ~write_track_descriptor_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class write_track_descriptor_process

} // end namespace

#endif // _KWIVER_WRITE_TRACK_DESCRIPTOR_PROCESS_H
