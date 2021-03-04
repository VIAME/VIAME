// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for read_object_track process
 */

#ifndef _KWIVER_READ_OBJECT_TRACK_PROCESS_H
#define _KWIVER_READ_OBJECT_TRACK_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// -------------------------------------------------------------------------------
/**
 * \class read_object_track_process
 *
 * \brief Reads a series or single set of track descriptors
 *
 * \iports
 * \iport{image_name}
 * \oport{track descriptor_set}
 */
class KWIVER_PROCESSES_NO_EXPORT read_object_track_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "read_object_track",
               "Reads object track sets from an input file." )

  read_object_track_process( kwiver::vital::config_block_sptr const& config );
  virtual ~read_object_track_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class read_object_track_process

} // end namespace

#endif // _KWIVER_READ_OBJECT_TRACK_PROCESS_H
