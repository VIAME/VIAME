/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
