/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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


#ifndef _KWIVER_SMQTK_DESCRIPTOR_H
#define _KWIVER_SMQTK_DESCRIPTOR_H

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

namespace kwiver {

// -----------------------------------------------------------------
/**
 * @brief SMQTK Descriptor Wrapper.
 *
 * This class implements a synchronous interface to a pipelined
 * implementation of a SMQTK descriptor.
 */
class SMQTK_Descriptor
{
public:
  // -- CONSTRUCTORS --
  SMQTK_Descriptor();
  ~SMQTK_Descriptor();

  /**
   * @brief Apply descriptor to image.
   *
   * This function takes an image as input and applies the SMQTK
   * descriptor. The resulting descriptor vector is returned.
   *
   * @param cv_img OCV image.
   * @param config_file Name of the descriptor configuration file.
   *
   * @return Calculated descriptor as vector of doubles.
   */
  std::vector< double > ExtractSMQTK(  cv::Mat cv_img, std::string const& config_file );

private:
  class priv;
  const std::unique_ptr<priv> d;


}; // end class SMQTK_Descriptor

} // end namespace

#endif /* _KWIVER_SMQTK_DESCRIPTOR_H */
