// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface and implementation of CPU timer classes
 */

#ifndef KWIVER_VITAL_CPU_TIMER_H
#define KWIVER_VITAL_CPU_TIMER_H

#include <vital/util/timer.h>

#include <string>
#include <ctime>
#include <iostream>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Interval timer class.
 *
 * This class represents an interval timer that measures CPU time. The
 * time reported is summed over all cores/threads active, so the
 * resulting time could be greater than wall clock time.

 */
class cpu_timer
  : public timer
{
public:
  cpu_timer()
    : m_start(0),
      m_end(0)
  { }

  ~cpu_timer()
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
    m_start = std::clock();
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
    m_end = std::clock();
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
      std::clock_t elapsed_seconds = std::clock() - m_start;
      return static_cast< double >(elapsed_seconds) / CLOCKS_PER_SEC;
    }
    else
    {
      std::clock_t elapsed_seconds = m_end - m_start;
      return static_cast< double >(elapsed_seconds) / CLOCKS_PER_SEC;
    }
  }

private:

  std::clock_t m_start;
  std::clock_t m_end;

}; // end class cpu_timer

// instantiate scoped timer
template class scoped_timer< cpu_timer >;
typedef scoped_timer< cpu_timer > scoped_cpu_timer;

} }   // end namespace

#endif /* KWIVER_VITAL_CPU_TIMER_H */
