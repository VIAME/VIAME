// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief extract_descriptors algorithm definition

#ifndef VITAL_ALGO_EXTRACT_DESCRIPTORS_H_
#define VITAL_ALGO_EXTRACT_DESCRIPTORS_H_

#include <viame/algorithm_framework/vital_config.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/feature_set.h>
#include <viame/core_types/image_container.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for extracting feature descriptors
class VITAL_ALGO_EXPORT extract_descriptors
  : public kwiver::vital::algorithm
{
public:
  extract_descriptors();
  PLUGGABLE_INTERFACE( extract_descriptors );
  /// Extract from the image a descriptor corresoponding to each feature
  ///
  /// \param [in]     image_data contains the image data to process
  /// \param [in,out] features the feature locations at which descriptors
  ///                 are extracted (may be modified).
  /// \param [in]     image_mask Mask image of the same dimensions as
  ///                            \p image_data where positive values indicate
  ///                            regions of \p image_data to consider.
  /// \returns a set of feature descriptors
  ///
  /// \note The feature_set passed into this function may modified to
  ///       reorder, remove, or duplicate some features to align with the
  ///       set of descriptors detected.  If the feature_set needs to change,
  ///       a new feature_set is created and returned by reference.

  virtual kwiver::vital::descriptor_set_sptr
  extract(
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::feature_set_sptr& features,
    kwiver::vital::image_container_sptr image_mask =
    kwiver::vital::image_container_sptr() ) const = 0;
};

/// Shared pointer for base extract_descriptors algorithm definition class
typedef std::shared_ptr< extract_descriptors > extract_descriptors_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_EXTRACT_DESCRIPTORS_H_
