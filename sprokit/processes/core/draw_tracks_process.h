// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Image display process interface.
 */

#ifndef _KWIVER_DRAW_TRACKS_PROCESS_H
#define _KWIVER_DRAW_TRACKS_PROCESS_H

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * @brief Display images
 *
 */
class KWIVER_PROCESSES_NO_EXPORT draw_tracks_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "draw_tracks",
               "Draw feature tracks on image." )

  // -- CONSTRUCTORS --
  draw_tracks_process( kwiver::vital::config_block_sptr const& config );
  virtual ~draw_tracks_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class draw_tracks_process

} // end namespace

#endif /* _KWIVER_DRAW_TRACKS_PROCESS_H */
