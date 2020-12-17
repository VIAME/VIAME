// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to abstract filter image algorithm
 */

#ifndef VITAL_ALGO_IMAGE_FILTER_H
#define VITAL_ALGO_IMAGE_FILTER_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital {
namespace algo {

/// \brief Abstract base class for image set filter algorithms.
/**
 * This interface supports arrows/algorithms that do a pixel by pixel
 * image modification, such as image enhancement. The resultant image
 * must be the same size as the input image.
 */
class VITAL_ALGO_EXPORT image_filter
  : public kwiver::vital::algorithm_def<image_filter>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "image_filter"; }

  /// Filter a  input image and return resulting image
  /**
   * This method implements the filtering operation. The method does
   * not modify the image in place. The resulting image must be a
   * newly allocated image which is the same size as the input image.
   *
   * \param image_data Image to filter.
   * \returns a filtered version of the input image
   */
  virtual kwiver::vital::image_container_sptr filter( kwiver::vital::image_container_sptr image_data ) = 0;

protected:
  image_filter();

};

/// type definition for shared pointer to a image_filter algorithm
typedef std::shared_ptr<image_filter> image_filter_sptr;

} } } // end namespace

#endif /* VITAL_ALGO_IMAGE_FILTER_H */
