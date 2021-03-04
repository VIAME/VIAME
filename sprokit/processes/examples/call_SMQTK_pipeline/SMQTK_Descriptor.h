// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
