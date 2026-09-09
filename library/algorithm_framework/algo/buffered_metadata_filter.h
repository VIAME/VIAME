// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Interface to abstract filter metadata algorithm.

#ifndef VITAL_ALGO_BUFFERED_METADATA_FILTER_H
#define VITAL_ALGO_BUFFERED_METADATA_FILTER_H

#include <viame/algorithm_framework/algo/algorithm.h>

#include <viame/algorithm_framework/algorithm_capabilities.h>

#include <viame/core_types/image_container.h>
#include <viame/core_types/metadata.h>

#include <viame/algorithm_framework/vital_config.h>

namespace kwiver {

namespace vital {

namespace algo {

/// Abstract base class for buffered metadata filter algorithms.
///
/// This interface supports arrows/algorithms that modify image metadata and
/// require some amount of "lookahead" to do so.
class VITAL_ALGO_EXPORT buffered_metadata_filter
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

  buffered_metadata_filter();
  PLUGGABLE_INTERFACE( buffered_metadata_filter );

  /// Provide one frame of metadata to the filter.
  ///
  /// This method implements the filtering operation, which may delay producing
  /// output until more frames are sent. The method does not modify the
  /// metadata in place.
  ///
  /// \param input_metadata Metadata to filter.
  /// \param input_image Image associated with the metadata (may be null).
  ///
  /// \returns The number of frames of output available.
  virtual size_t send(
    metadata_vector const& input_metadata,
    image_container_scptr const& input_image ) = 0;

  /// Return one frame of processed metadata.
  ///
  /// \throw std::runtime_error If no output is available.
  virtual metadata_vector receive() = 0;

  /// Force the buffer to process all sent input immediately.
  ///
  /// This method forces a wipe of all internal input buffers, ensuring that
  /// immediately subsequent calls to \c unavailable_frames() return zero. This
  /// may result in frames being processed in an inferior manner. This method
  /// should be called when there is no more input.
  virtual size_t flush() = 0;

  /// Return the number of processed frames.
  virtual size_t available_frames() const = 0;

  /// Return the number of yet-unprocessed frames.
  virtual size_t unavailable_frames() const = 0;

  /// Return capabilities of concrete implementation.
  algorithm_capabilities const& get_implementation_capabilities() const;

protected:
  void set_capability(
    algorithm_capabilities::capability_name_t const& name, bool value );

private:
  algorithm_capabilities m_capabilities;
};

using buffered_metadata_filter_sptr =
  std::shared_ptr< buffered_metadata_filter >;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif
