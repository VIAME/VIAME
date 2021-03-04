// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Image display process interface.
 */

#ifndef _KWIVER_IMAGE_VIEWER_PROCESS_H
#define _KWIVER_IMAGE_VIEWER_PROCESS_H

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_ocv_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * @brief Display images
 *
 */
class KWIVER_PROCESSES_OCV_NO_EXPORT image_viewer_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "image_viewer",
               "Display input image and delay" )

  // -- CONSTRUCTORS --
  image_viewer_process( kwiver::vital::config_block_sptr const& config );
  virtual ~image_viewer_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class image_viewer_process

} // end namespace

#endif // _KWIVER_IMAGE_VIEWER_PROCESS_H
