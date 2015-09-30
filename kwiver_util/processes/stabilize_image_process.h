/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_STABILIZE_IMAGE_PROCESS_H_
#define _KWIVER_STABILIZE_IMAGE_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <boost/scoped_ptr.hpp>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class stabilize_image_process
 *
 * \brief Stabilizes a series of image.
 *
 * \iports
 * \iport{timestamp}
 * \iport{image}
 *
 * \oports
 * \oport{src_to_ref_homography}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT stabilize_image_process
  : public sprokit::process
{
  public:
  stabilize_image_process( kwiver::vital::config_block_sptr const& config );
  virtual ~stabilize_image_process();

  protected:
    virtual void _configure();
    virtual void _step();

  private:
    void make_ports();
    void make_config();


    class priv;
    boost::scoped_ptr<priv> d;
 }; // end class stabilize_image_process


} // end namespace
#endif /* _KWIVER_STABILIZE_IMAGE_PROCESS_H_ */
