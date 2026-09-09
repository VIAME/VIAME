// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface to abstract filter metadata algorithm

#ifndef VITAL_ALGO_METADATA_FILTER_H
#define VITAL_ALGO_METADATA_FILTER_H

#include <viame/algorithm_framework/algo/algorithm.h>

#include <viame/algorithm_framework/algorithm_capabilities.h>

#include <viame/core_types/image_container.h>
#include <viame/core_types/metadata.h>

#include <viame/algorithm_framework/vital_config.h>

namespace kwiver {

namespace vital {

namespace algo {

/// Abstract base class for metadata filter algorithms.
///
/// This interface supports arrows/algorithms that modify image metadata.
class VITAL_ALGO_EXPORT metadata_filter
  : public kwiver::vital::algorithm
{
public:
  /// Algorithm can use the frame image for its operation.
  ///
  /// This capability indicates if the algorithm is able to make use of the
  /// frame image. If this is not set, it implies that passing a null pointer
  /// as the input image to #filter will not affect the results, which may
  /// afford significant optimization opportunities to users.
  static const algorithm_capabilities::capability_name_t CAN_USE_FRAME_IMAGE;

  metadata_filter();
  PLUGGABLE_INTERFACE( metadata_filter );

  /// Filter metadata and return resulting metadata.
  ///
  /// This method implements the filtering operation. The method does not
  /// modify the metadata in place.
  ///
  /// \param input_metadata Metadata to filter.
  /// \param input_image Image associated with the metadata (may be null).
  ///
  /// \returns Filtered version of the input metadata.
  virtual kwiver::vital::metadata_vector filter(
    kwiver::vital::metadata_vector const& input_metadata,
    kwiver::vital::image_container_scptr const& input_image ) = 0;

  /// Return capabilities of concrete implementation.
  ///
  /// This method returns the capabilities of the algorithm implementation.
  ///
  /// \return Reference to supported algorithm capabilities.
  algorithm_capabilities const& get_implementation_capabilities() const;

protected:
  void set_capability(
    algorithm_capabilities::capability_name_t const& name, bool value );

private:
  algorithm_capabilities m_capabilities;
};

/// Type alias for shared pointer to a metadata_filter algorithm.
using metadata_filter_sptr = std::shared_ptr< metadata_filter >;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif
