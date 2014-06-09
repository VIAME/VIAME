/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_TIMESTAMP_H_
#define _KWIVER_TIMESTAMP_H_

#include <stdint.h>
#include <ostream>
#include <istream>

namespace kwiver
{
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
class timestamp
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

private:
  bool m_valid_time; ///< indicates valid timestamp
  bool m_valid_frame;

  time_t m_time; ///< frame time in seconds
  frame_t  m_frame;
}; // end class timestamp

  std::ostream& operator<< ( std::ostream& str, timestamp const& obj );


  /**
   * \brief Input operator
   *
   * Convert object from string representation to native form.  This
   * is primarily used for object specific behaviour for
   * boost:lexical_cast when supplying default values for static
   * ports.
   *
   * The expected format for the string representation of a timestamp
   * is:
   *
   * <frame> <time in seconds>
   *
   * For example:
   \code
   300 10.0
   \endcode
   *
   * @param str input stream
   * @param obj string to read from
   *
   * @return
   */
  std::istream& operator>> ( std::istream& str, timestamp& obj );

} // end namespace

#endif /* _KWIVER_TIMESTAMP_H_ */
