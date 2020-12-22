// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface and implementation of WALL timer classes
 */

#ifndef KWIVER_VITAL_WALL_TIMER_H
#define KWIVER_VITAL_WALL_TIMER_H

#include <vital/util/timer.h>

#include <string>
#include <chrono>
#include <iostream>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Interval wall clock timer class.
 *
 * This class represents an interval timer that measures wall clock time.
 */
class wall_timer
  : public timer
{
public:
  wall_timer()
  { }

  ~wall_timer()
  {
    m_active = false;
  }

  virtual  bool timer_available() { return true; }

  /**
   * @brief Start the timer.
   *
   * The clock time when this timer is started is saved.
   */
  virtual void start()
  {
    m_active = true;
    m_start = std::chrono::steady_clock::now();
  }

  /**
   * @brief Stop the timer.
   *
   * The time this clock was stopped is saved. This value is used to
   * determine the elapsed time.
   */
  virtual void stop()
  {
    m_active = false;
    m_end = std::chrono::steady_clock::now();
  }

  /**
   * @brief Calculate elapsed time.
   *
   * The elapsed time of this timer is returned. This method works if
   * the timer is stopped or still running.
   *
   * @return Seconds since the timer was started.
   */
  virtual double elapsed() const
  {
    if (m_active)
    {
      // Take a snapshot of the interval.
      std::chrono::duration< double > elapsed_seconds = std::chrono::steady_clock::now() - m_start;
      return elapsed_seconds.count();
    }
    else
    {
      std::chrono::duration< double > elapsed_seconds = m_end - m_start;
      return elapsed_seconds.count();
    }
  }

private:

  std::chrono::time_point< std::chrono::steady_clock > m_start;
  std::chrono::time_point< std::chrono::steady_clock > m_end;

}; // end class wall_timer

template class scoped_timer< wall_timer >;
typedef scoped_timer< wall_timer > scoped_wall_timer;

} }   // end namespace

#endif /* KWIVER_VITAL_SCOPED_TIMER_H */
