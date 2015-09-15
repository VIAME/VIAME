/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_
#define _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <boost/scoped_ptr.hpp>

namespace kwiver
{

/*!
 * \class kw_archive_index_writer
 *
 * \brief KW archive writer process
 *
 * \process Writes kw video archive
 *
 * This process writes a multi-file KW archive of the image stream
 * received.  The archive directory
 *
 * \iports
 * \iport{timestamp}
 * Time associated with current frame
 *
 * \iport{image}
 * Image frame from video being processed
 *
 * \iport{src_to_ref_homography}
 * Homography that will transform current image to reference coordinates
 *
 * \iport{corner_points}
 * Corner points for image in lat/lon coordinates
 *
 * \iport{gsd}
 * Scaling of the image in meters per pixel.
 */

class KWIVER_PROCESSES_NO_EXPORT kw_archive_writer_process
  : public sprokit::process
{
public:

  /**
   * \brief Constructor
   *
   * @param config Process config object
   *
   * @return
   */
  kw_archive_writer_process( kwiver::vital::config_block_sptr const& config );

  /**
   * \brief Destructor
   */
  ~kw_archive_writer_process();

protected:
  void _configure();
  void _init();
  void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  boost::scoped_ptr<priv> d;
};

} // end namespace kwiver

#endif /* _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_ */
