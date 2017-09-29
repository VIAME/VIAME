/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
