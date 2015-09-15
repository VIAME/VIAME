/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_FRAME_LIST_PROCESS_H_
#define _KWIVER_FRAME_LIST_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <boost/scoped_ptr.hpp>

namespace kwiver
{

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
  frame_list_process( kwiver::vital::config_block_sptr const& config );
  virtual ~frame_list_process();


protected:
  void _configure();
  void _init();
  void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  boost::scoped_ptr<priv> d;
}; // end class frame_list_process

}  // end namespace

#endif /* _KWIVER_FRAME_LIST_PROCESS_H_ */
