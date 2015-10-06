/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_COMPUTE_HOMOGRAPHY_PROCESS_H_
#define _KWIVER_COMPUTE_HOMOGRAPHY_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class compute_homography_process
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
class KWIVER_PROCESSES_NO_EXPORT compute_homography_process
  : public sprokit::process
{
  public:
  compute_homography_process( kwiver::vital::config_block_sptr const& config );
  virtual ~compute_homography_process();

  protected:
    virtual void _configure();
    virtual void _step();

  private:
    void make_ports();
    void make_config();


    class priv;
    const std::unique_ptr<priv> d;
 }; // end class compute_homography_process


} // end namespace
#endif /* _KWIVER_COMPUTE_HOMOGRAPHY_PROCESS_H_ */
