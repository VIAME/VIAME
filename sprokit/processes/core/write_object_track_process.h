// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for object_track_set_output process
 */

#ifndef _KWIVER_WRITE_OBJECT_TRACK_PROCESS_H
#define _KWIVER_WRITE_OBJECT_TRACK_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

  // -----------------------------------------------------------------------------
/**
 * \class write_object_track_process
 *
 * \brief Write a set of track descriptors to a file
 *
 * \iports
 * \iport{track descriptor_set}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT write_object_track_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "write_object_track",
               "Writes object track sets to an output file. "
               "All descriptors are written to the same file." )

  write_object_track_process( kwiver::vital::config_block_sptr const& config );
  virtual ~write_object_track_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class write_object_track_process

} // end namespace

#endif // _KWIVER_WRITE_OBJECT_TRACK_PROCESS_H
