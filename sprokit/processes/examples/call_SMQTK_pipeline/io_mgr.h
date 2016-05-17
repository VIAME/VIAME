/*ckwg +29
 * Copyright 2015-2016 by Kitware, Inc.
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

#ifndef _KWIVER_IO_MGR_H
#define _KWIVER_IO_MGR_H

#include "smqtk_extract_export.h"
#include <arrows/processes/kwiver_type_traits.h>

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>


namespace kwiver {

// -----------------------------------------------------------------
/**
 * @brief Class to manage input and output from endcaps.
 *
 * This class is implemented as a singleton so that both endcaps and
 * the pipeline control method can share this class.
 *
 * Currently this class expecte to handle only one image. If more than
 * one image is needed, then some queueing will be needed.
 */

class SMQTK_EXTRACT_EXPORT io_mgr
{
public:
  // -- CONSTRUCTORS --
  io_mgr() { }
  virtual ~io_mgr() { }

  //
  static io_mgr* Instance();

  // -- ACCESSORS --
  cv::Mat const& GetImage() const { return m_image; }
  kwiver::vital::double_vector_sptr GetDescriptor() const { return m_descriptor; }

  // -- MANIPULATORS --
  void SetImage( cv::Mat const& img ) { m_image = img; }
  void SetDescriptor( kwiver::vital::double_vector_sptr vec) { m_descriptor = vec; }

private:
  // input image
  cv::Mat m_image;

  // output descriptor vector
  kwiver::vital::double_vector_sptr m_descriptor;

  static io_mgr* s_instance;
}; // end class io_mgr

} // end namespace

#endif /* _KWIVER_IO_MGR_H */
