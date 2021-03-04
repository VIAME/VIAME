// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining abstract image object detector
 */

#ifndef VITAL_ALGO_IMAGE_OBJECT_DETECTOR_H_
#define VITAL_ALGO_IMAGE_OBJECT_DETECTOR_H_

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------
/**
 * @brief Image object detector base class/
 *
 */
class VITAL_ALGO_EXPORT image_object_detector
: public algorithm_def<image_object_detector>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "image_object_detector"; }

  /// Find all objects on the provided image
  /**
   * This method analyzes the supplied image and along with any saved
   * context, returns a vector of detected image objects.
   *
   * \param image_data the image pixels
   * \returns vector of image objects found
   */
  virtual detected_object_set_sptr
      detect( image_container_sptr image_data ) const = 0;


  /// Find all objects on the provided images
  /**
   * This method analyzes the supplied images and along with any saved
   * context, returns a vector of detected image objects.
   *
   * By default this just calls the single image detector, but this
   * call can be over-written for detectors which perform optimized
   * batching.
   *
   * \param images the image pixels
   * \returns vector of image objects found
   */
  virtual std::vector< vital::detected_object_set_sptr >
      batch_detect( const std::vector< image_container_sptr >& images ) const;

protected:
  image_object_detector();
};

/// Shared pointer for generic image_object_detector definition type.
typedef std::shared_ptr< image_object_detector > image_object_detector_sptr;

} } } // end namespace

#endif //VITAL_ALGO_IMAGE_OBJECT_DETECTOR_H_
