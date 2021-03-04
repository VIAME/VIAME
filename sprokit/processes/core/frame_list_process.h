// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_FRAME_LIST_PROCESS_H_
#define _KWIVER_FRAME_LIST_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

namespace kwiver {

// ----------------------------------------------------------------
/**
 * \class frame_list_process
 *
 * \brief Reads a series of images
 *
 * \oports
 * \oport{image}
 *
 * \oport{frame}
 * \oport{time}
 */
class KWIVER_PROCESSES_NO_EXPORT frame_list_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "frame_list_input",
               "Reads a list of image file names and generates stream of "
               "images and associated time stamps." )

  frame_list_process( kwiver::vital::config_block_sptr const& config );
  virtual ~frame_list_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class frame_list_process

}  // end namespace

#endif /* _KWIVER_FRAME_LIST_PROCESS_H_ */
