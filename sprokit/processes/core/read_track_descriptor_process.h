// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for read_track_descriptor process
 */

#ifndef _KWIVER_READ_TRACK_DESCRIPTOR_PROCESS_H
#define _KWIVER_READ_TRACK_DESCRIPTOR_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// -------------------------------------------------------------------------------
/**
 * \class read_track_descriptor_process
 *
 * \brief Reads a series or single set of track descriptors
 *
 * \iports
 * \iport{image_name}
 * \oport{track descriptor_set}
 */
class KWIVER_PROCESSES_NO_EXPORT read_track_descriptor_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "read_track_descriptor",
               "Reads track descriptor sets from an input file." )

  read_track_descriptor_process( kwiver::vital::config_block_sptr const& config );
  virtual ~read_track_descriptor_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class read_track_descriptor_process

} // end namespace

#endif // _KWIVER_READ_TRACK_DESCRIPTOR_PROCESS_H
