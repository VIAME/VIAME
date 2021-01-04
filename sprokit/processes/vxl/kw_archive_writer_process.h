// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_
#define _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_vxl_export.h"

#include <memory>

namespace kwiver
{

/*!
 * \class kw_archive_writer_process
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
 *
 * \iport{stream_id}
 * Optional input stream ID to put in the KWA file.
 *
 * \iport{filename}
 * Optional input filename (no extension) to write the KWA to.
 */

class KWIVER_PROCESSES_VXL_NO_EXPORT kw_archive_writer_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "kw_archive_writer",
               "Writes kw archives." )

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
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace kwiver

#endif /* _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_ */
