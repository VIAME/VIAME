// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Interface for video_output

#ifndef VITAL_ALGO_VIDEO_OUTPUT_H_
#define VITAL_ALGO_VIDEO_OUTPUT_H_

#include <viame/algorithm_framework/algo/vital_algo_export.h>

#include <viame/algorithm_framework/algo/algorithm.h>

#include <viame/algorithm_framework/algorithm_capabilities.h>

#include <viame/core_types/image_container.h>
#include <viame/core_types/metadata.h>
#include <viame/core_types/metadata_map.h>
#include <viame/core_types/timestamp.h>
#include <viame/core_types/video_raw_image.h>
#include <viame/core_types/video_raw_metadata.h>
#include <viame/core_types/video_settings.h>
#include <viame/core_types/video_uninterpreted_data.h>

#include <viame/algorithm_framework/vital_config.h>

#include <string>
#include <vector>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for writing videos.
///
/// This class represents an abstract interface for writing videos. Once the
/// video is opened, frames may be added in order.
class VITAL_ALGO_EXPORT video_output
  : public kwiver::vital::algorithm
{
public:
  /// Writer supports a global frame rate.
  ///
  /// This capability indicates if the implementation is able to write a video
  /// which specifies a global, constant frame rate.
  static const algorithm_capabilities::capability_name_t SUPPORTS_FRAME_RATE;

  /// Writer can write per-frame time codes.
  ///
  /// This capability indicates if the implementation is able to write distinct
  /// time codes for each frame. Note that writers without this capability may
  /// still use the frame time for error checking, insertion of "blank" frames,
  /// or synchronization with metadata.
  static const algorithm_capabilities::capability_name_t SUPPORTS_FRAME_TIME;

  /// Writer can write metadata.
  ///
  /// This capability indicates if the implementation is able to write
  /// metadata.
  static const algorithm_capabilities::capability_name_t SUPPORTS_METADATA;

  /// Writer can write uninterpreted data.
  ///
  /// This capability indicates if the implementation can take data which a
  /// video input did not interpret and put it back in the video stream.
  static const algorithm_capabilities::capability_name_t
    SUPPORTS_UNINTERPRETED_DATA;

  video_output();
  PLUGGABLE_INTERFACE( video_output );

  /// Open a video stream.
  ///
  /// This method opens the specified video stream for writing. The format of
  /// the name depends on the concrete implementation. It could be a file name,
  /// a directory, or a URI.
  ///
  /// \param video_name Identifier of the video stream.
  /// \param settings
  ///   Additional information used to configure the video output.
  ///
  /// \throws exception Thrown if opening the video stream failed.
  virtual void open(
    std::string video_name,
    video_settings const* settings ) = 0;

  /// Close video stream.
  ///
  /// Close the currently opened stream and release resources. Closing a stream
  /// that is already closed does not cause a problem.
  virtual void close() = 0;

  /// Check whether state of video stream is good.
  ///
  /// This method checks the current state of the video stream to see if it is
  /// good. A stream is good if it is ready to receive images and/or metadata.
  ///
  /// \return \c true if video stream is good, \c false if not good.
  virtual bool good() const = 0;

  /// Add a frame image to the video stream.
  ///
  /// This method writes the next frame image to the video stream. The
  /// timestamp should be greater than that of the previously written frame, as
  /// many implementations are unable to write frames out of order.
  ///
  /// \throws video_stream_exception
  ///   Thrown if is an error in the video stream.
  virtual void add_image(
    kwiver::vital::image_container_sptr const& image,
    kwiver::vital::timestamp const& ts ) = 0;

  /// Add a raw frame image to the video stream.
  ///
  /// This method writes the raw image to the video stream. There is no
  /// guarantee that this functions correctly when intermixed with non-raw
  /// images.
  virtual void add_image( video_raw_image const& image );

  /// Add metadata collection to the video stream.
  ///
  /// This method adds metadata to the video stream. Depending on the
  /// implementation, the metadata may be written immediately, or may be
  /// deferred until the next frame is written. For this reason, the metadata's
  /// timestamp should be greater than that of the previously written frame.
  ///
  /// For implementations that do not support metadata, this method does
  /// nothing.
  ///
  /// \throws video_stream_exception
  ///   Thrown if is an error in the video stream.
  virtual void add_metadata( kwiver::vital::metadata const& md ) = 0;

  /// Add a frame of raw metadata to the video stream.
  ///
  /// This method writes the raw metadata to the video stream. There is no
  /// guarantee that this functions correctly when intermixed with non-raw
  /// metadata.
  virtual void add_metadata( video_raw_metadata const& md );

  /// Add a frame of uninterpreted data to the video stream.
  ///
  /// This method writes the uninterpreted data to the video stream.
  virtual void add_uninterpreted_data(
    video_uninterpreted_data const& misc_data );

  /// Extract implementation-specific video encoding settings.
  ///
  /// The returned structure is intended to be passed to a video encoder of
  /// similar implementation to produce similarly formatted output. The
  /// returned value may not be identical to the one passed to this object via
  /// open().
  ///
  /// \return Implementation video settings, or \c nullptr if none are needed.
  virtual vital::video_settings_sptr implementation_settings() const;

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

/// Shared pointer type for generic video_output definition type.
using video_output_sptr = std::shared_ptr< video_output >;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif
