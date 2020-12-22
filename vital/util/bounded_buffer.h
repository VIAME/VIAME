// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface and implementation for bounded buffer
 */

#ifndef KWIVER_VITAL_UTIL_BOUNDED_BUFFER_H_
#define KWIVER_VITAL_UTIL_BOUNDED_BUFFER_H_

#include <condition_variable> //+ c++11 required
#include <mutex>
#include <vector>
#include <stdexcept>

#include <vital/noncopyable.h>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Simple bounded buffer.
 *
 * This class represents a fixed size bounded buffer designed for
 * communication between threads.
 */
template <class T>
class bounded_buffer : private vital::noncopyable
{
public:
  typedef std::unique_lock<std::mutex> lock;

  bounded_buffer( int size )
  : begin(0),
    end(0),
    buffered(0),
    circular_buf( size )
  {
    if (size < 1)
    {
      throw std::length_error( "Invalid length specified for bounded_buffer. Must be greater than zero.");
    }
  }

  /**
   * @brief Reset the contents of the buffer to empty.
   *
   * This is a little abrupt, so the caller must ensure that there is
   * nothing of value in the buffer.
   */
  void Reset()
  {
    lock lk(monitor);
    begin = 0;
    end = 0;
    buffered = 0;

    // just in case
    buffer_not_full.notify_one();
  }

  /**
   * @brief Send element to buffer.
   *
   * The specified element is added (copied) to the end of the bounded
   * buffer. The calling process will wait if there is no available
   * space.
   *
   * @param[in] m Element to add to buffer.
   */
  void Send (T const& m)
  {
    lock lk(monitor);
    while (buffered == circular_buf.size())
    {
      buffer_not_full.wait(lk);
    }

    circular_buf[end] = m;
    end = (end+1) % circular_buf.size();
    ++buffered;

    buffer_not_empty.notify_one();
  }

  /**
   * @brief Receive element from buffer.
   *
   * The oldest element in the buffer is returned to the caller. The
   * calling thread will wait if the buffer is empty.
   *
   * @return The oldest
   */
  T Receive()
  {
    lock lk(monitor);
    while (buffered == 0)
    {
      buffer_not_empty.wait(lk);
    }

    T i = circular_buf[begin];
    begin = (begin+1) % circular_buf.size();
    --buffered;

    buffer_not_full.notify_one();
    return i;
  }

  /**
   * @brief Test if buffer is empty.
   *
   * This method indicates if the buffer is empty.  A polling approach
   * can be implemented using this method.
   *
   * @return \b true if buffer is empty. \b false if not empty.
   */
  bool Empty() const
  {
    return (buffered == 0);
  }

  /**
   * @brief Test if buffer is full.
   *
   * This method indicates if the buffer is full. A polling approach
   * can be implemented using this method.
   *
   * @return
   */
  bool Full() const
  {
    return  (buffered == circular_buf.size());
  }

private:
  size_t begin, end, buffered;
  std::vector< T > circular_buf;
  std::condition_variable buffer_not_full, buffer_not_empty;
  std::mutex monitor;
};

} } // end namespace

#endif
