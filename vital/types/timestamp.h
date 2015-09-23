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

#ifndef _VITAL_TIMESTAMP_H_
#define _VITAL_TIMESTAMP_H_

#include <stdint.h>
#include <ostream>
#include <istream>

#include <vital/vital_export.h>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * \brief Frame time.
 *
 * This class represents a timestamp for a single video frame.  The
 * time is represented in seconds and frame numbers start at one.
 *
 * A timestamp has the notion of valid time and valid frame. This is
 * useful when dealing with interpolated timestamps. In this case, a
 * timestamps may have a time, but no frame.
 */
class VITAL_EXPORT timestamp
{
public:
  // -- TYPES --
  typedef double time_t;
  typedef int64_t frame_t;

  /**
   * \brief Default constructor.
   *
   * Created an invalid timestamp.
   */
  timestamp();


  /**
   * \brief Constructor
   *
   * Creates a valid timestamp with specified time and frame number.
   *
   * @param t Time for timestamp
   * @param f Frame number for timestamp
   */
  explicit timestamp( time_t t, frame_t f);

  /**
   * \brief Is timestamp valid.
   *
   * Both the time and frame must be set for a timestamp to be totally
   * valid.
   *
   * @return \b true if both time and frame are valid
   */
  bool is_valid() const { return m_valid_time && m_valid_frame; }

  /**
   * \brief Timestamp has valid time.
   *
   * Indicates that the time has been set for this timestamp.
   *
   * @return \b true if time has been set
   */
  bool has_valid_time() const { return m_valid_time; }


  /**
   * \brief Timestamp has valid frame number.
   *
   * Indicates that the frame number has been set for this timestamp.
   *
   * @return \b true if frame number has been set
   */
  bool has_valid_frame() const { return m_valid_frame; }


  /**
   * \brief Get time from timestamp.
   *
   * The time portion of the timestamp is returned in seconds. The
   * value will be undetermined if the timestamp does not have a valid time.
   * \sa has_valid_time()
   *
   * @return Frame time in seconds
   */
  time_t get_time() const { return m_time; }

  /**
   * \brief Get time in seconds.
   *
   * The time portion of the timestamp is returned in seconds and fractions.
   *
   * \return time in seconds.
   */
  double get_time_seconds() const;

  /**
   * \brief Get frame number from timestamp.
   *
   * The frame number value from the timestamp is returned. The first
   * frame in a sequence is usually one. The frame number will be
   * undetermined if the timestamp does not have a valid frame number
   * set.
   * \sa has_valid_frame()
   *
   * @return Frame number.
   */
  frame_t get_frame() const { return m_frame; }


  /**
   * \brief Set time portion of timestamp.
   *
   * @param t Time for frame.
   */
  timestamp& set_time( time_t t );


  /**
   * \brief Set frame portion of timestamp.
   *
   * @param f Frame number
   */
  timestamp& set_frame( frame_t f);


  /**
   * \brief Set timestamp totally invalid.
   *
   * Both the frame and time are set to
   */
  timestamp& set_invalid();


  /**
   * \brief Format object in a readable manner.
   *
   * This method formats a time stamp in a readable and recognizable
   * manner suitable form debugging and logging.
   *
   * @return formatted timestamp
   */
  std::string pretty_print() const;


private:
  bool m_valid_time; ///< indicates valid timestamp
  bool m_valid_frame;

  time_t m_time; ///< frame time in seconds
  frame_t  m_frame;

  ///\todo Convert time to  int64 microseconds, add unique field to denote reference.
  /// The concept is to prevent comparing timestamps that have a different reference
  /// (e.g. from a different video stream or having a different start time offset).

}; // end class timestamp


inline std::ostream& operator<< ( std::ostream& str, timestamp const& obj )
{ str << obj.pretty_print(); return str; }

} } // end namespace

#endif /* _VITAL_TIMESTAMP_H_ */
