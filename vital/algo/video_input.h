/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

/**
 * \file
 * \brief Interface for video_input
 */

#ifndef VITAL_ALGO_VIDEO_INPUT_H_
#define VITAL_ALGO_VIDEO_INPUT_H_

#include <vital/vital_config.h>
#include <vital/algo/vital_algo_export.h>

#include <vital/algorithm_capabilities.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/metadata.h>
#include <vital/types/timestamp.h>

#include <string>
#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

// ==================================================================
/// An abstract base class for reading videos
/**
 * This class represents an abstract interface for reading
 * videos. Once the video is opened, the frames are returned in order.
 *
 * Use cases:
 * ----------
 *
 * 1) Reading video from a directory of images.
 *
 * 2) Reading video frames from a list of file names.
 *
 * 3) Reading video from mpeg/video file (one of many formats) (e.g. FMV)
 *
 * 4) Reading video from mpeg/video file (one of many formats) with
 *    cropping (e.g. WAMI). This includes Providing geostationary
 *    images by cropping to a specific region from an image. This may
 *    result in no data if the geo region and image do not intersect.
 *
 * 5) Reading video from network stream. (RTSP) This may result in
 *    unexpected end of video conditions and network related
 *    disruptions (e.g. missing frames, connection terminating, ...)
 *
 * A note about the basic capabilities:
 *
 * HAS_EOV - This capability is set to true if the video source can
 *     determine end of video. This is usually the case if the video
 *     is being read from a file, but may not be known if the video is
 *     coming from a streaming source.
 *
 * HAS_FRAME_NUMBERS - This capability is set to true if the video source
 *     supplies frame numbers. If the video source specifies a frame
 *     number, then that number is used when forming a time stamp. If
 *     the video does not supply a frame number, the time stamp will
 *     not have a frame number.
 *
 * HAS_FRAME_TIME - This capability is set to true if the video source
 *     supplies a frame time. If a frame time is supplied, it is made
 *     available in the time stamp for that frame. If the frame time
 *     is not supplied, then the timestamp will hot have the time set.
 *
 * HAS_FRAME_DATA - This capability is set to true if the video source
 *     supplies frame images. It may seem strange for a video input
 *     algorithm to not supply image data, but happens with a reader
 *     that only supplies the metadata.
 *
 * HAS_ABSOLUTE_FRAME_TIME - This capability is set to true if the
 *     video source supplies an absolute, rather than relative frame
 *     time. This capability is not set if an absolute frame time can
 *     not be found, or if the absolute frame time is configured as
 *     "none".
 *
 * HAS_METADATA - This capability is set if the video source supplies
 *     some type of metadata. The metadata could be in 0601 or 0104 data
 *     formats or a different source.
 *
 * HAS_TIMEOUT - This capability is set if the implementation supports the
 *     timeout parameter on the next_frame() method.
 *
 * All implementations \b must support the basic traits, in that they
 * are registered with a \b true or \b false value. Additional
 * implementation specific (extended) traits may be added. The
 * application should first check to see if a extended trait is
 * registered by calling has_trait() since the actual implementation
 * is set by a configuration entry and not directly known by the
 * application.
 *
 * Extended capabilities can be created to publish capabilities of
 * non-standard video sources. These capabilities should be namespaced using
 * the name (or abbreviation) of the concrete algorithm followed by
 * the abbreviation of the capability.
 */
class VITAL_ALGO_EXPORT video_input
  : public kwiver::vital::algorithm_def<video_input>
{
public:

  // Common capabilities
  // -- basic capabilities --
  static const algorithm_capabilities::capability_name_t HAS_EOV;         // has end of video indication
  static const algorithm_capabilities::capability_name_t HAS_FRAME_NUMBERS;
  static const algorithm_capabilities::capability_name_t HAS_FRAME_TIME;
  static const algorithm_capabilities::capability_name_t HAS_FRAME_DATA;
  static const algorithm_capabilities::capability_name_t HAS_FRAME_RATE;
  static const algorithm_capabilities::capability_name_t HAS_ABSOLUTE_FRAME_TIME;
  static const algorithm_capabilities::capability_name_t HAS_METADATA;
  static const algorithm_capabilities::capability_name_t HAS_TIMEOUT;

  virtual ~video_input();

  /// Return the name of this algorithm
  static std::string static_type_name() { return "video_input"; }


  /**
   * \brief Open a video stream.
   *
   * This method opens the specified video stream for reading. The
   * format of the name depends on the concrete implementation. It
   * could be a file name or it could be a URI.
   *
   * Capabilities are set in this call, so they are available after.
   *
   * \param video_name Identifier of the video stream.
   *
   * \note Once a video is opened, it starts in an invalid state
   * (i.e. before the first frame of video). You must call \c next_frame()
   * to step to the first frame of video before calling \c frame_image().
   *
   * \throws exception if open failed
   */
  virtual void open( std::string video_name ) = 0;


  /**
   * \brief Close video stream.
   *
   * Close the currently opened stream and release resources.  Closing
   * a stream that is already closed does not cause a problem.
   */
  virtual void close() = 0;


  /**
   * \brief Return end of video status.
   *
   * This method returns the end-of-video status of the input
   * video. \b true is returned if the last frame has been returned.
   *
   * This method will always return \b false for video streams that have
   * no ability to detect end of video, such as network streams.
   *
   * \return \b true if at end of video, \b false otherwise.
   */
  virtual bool end_of_video() const = 0;


  /**
   * \brief Check whether state of video stream is good.
   *
   * This method checks the current state of the video stream to see
   * if it is good. A stream is good if it refers to a valid frame
   * such that calls to \c frame_image() and \c frame_metadata()
   * are expected to return meaningful data.  After calling \c open()
   * the initial video state is not good until the first call to
   * \c next_frame().
   *
   * \return \b true if video stream is good, \b false if not good.
   */
  virtual bool good() const = 0; // like io stream API


  /**
   * \brief Advance to next frame in video stream.
   *
   * This method advances the video stream to the next frame, making
   * the image and metadata available. The returned timestamp is for
   * new current frame.
   *
   * The timestamp returned may be missing either frame number or time
   * or both, depending on the actual implementation.
   *
   * Calling this method will make a new image and metadata packets
   * available. They can be retrieved by calling frame_image() and
   * frame_metadata().
   *
   * Check the HAS_TIMEOUT capability from the concrete implementation to
   * see if the timeout feature is supported.
   *
   * If the video input is already an end, then calling this method
   * will return \b false.
   *
   * \param[out] ts Time stamp of new frame.
   * \param[in] timeout Number of seconds to wait. 0 = no timeout.
   *
   * \return \b true if frame returned, \b false if end of video.
   *
   * \throws video_input_timeout_exception when the timeout expires.
   * \throws video_stream_exception when there is an error in the video stream.
   */
  virtual bool next_frame( kwiver::vital::timestamp& ts,
                           uint32_t timeout = 0 ) = 0;


  /**
   * \brief Get current frame from video stream.
   *
   * This method returns the image from the current frame.  If the
   * video input is already an end, then calling this method will
   * return a null pointer.
   *
   * This method is idempotent. Calling it multiple times without
   * calling next_frame() will return the same image.
   *
   * \return Pointer to image container.
   *
   * \throws video_stream_exception when there is an error in the video stream.
   */
  virtual kwiver::vital::image_container_sptr frame_image( ) = 0;


  /**
   * \brief Get metadata collection for current frame.
   *
   * This method returns the metadata collection for the current
   * frame. It is best to call this after calling next_frame() to make
   * sure the metadata and video are synchronized and that no metadata
   * collections are lost.
   *
   * Metadata typically occurs less frequently than video frames, so
   * if you call next_frame() and frame_metadata() together while
   * processing a video, there may be times where no metadata is
   * returned. In this case an empty metadata vector will be returned.
   *
   * Also note that the metadata collection contains a timestamp that
   * can be used to determine where the metadata fits in the video
   * stream.
   *
   * In video streams without metadata (as determined by the stream
   * capability), this method may return and empty vector, indicating no
   * new metadata has been found.
   *
   * Calling this method at end of video will return an empty metadata
   * vector.
   *
   * This method is idempotent. Calling it multiple times without
   * calling next_frame() will return the same metadata.
   *
   * @return Vector of metadata pointers.
   *
   * \throws video_stream_exception when there is an error in the video stream.
   */
  virtual kwiver::vital::metadata_vector frame_metadata() = 0;


  /**
   * \brief Get frame rate from the video.
   *
   * If frame rate is not supported, return -1.
   *
   * \return Frame rate.
   */
  virtual double frame_rate();


  /**
   * \brief Return capabilities of concrete implementation.
   *
   * This method returns the capabilities for the currently opened
   * video.
   *
   * \return Reference to supported video capabilities.
   */
  algorithm_capabilities const& get_implementation_capabilities() const;


protected:
  video_input(); // CTOR

  void set_capability( algorithm_capabilities::capability_name_t const& name, bool val );

private:
  algorithm_capabilities m_capabilities;
};

/// Shared pointer type for generic video_input definition type.
typedef std::shared_ptr<video_input> video_input_sptr;

} } } // end namespace

#endif // VITAL_ALGO_VIDEO_INPUT_H_
