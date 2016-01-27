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
 * \brief Interface and implementation of WALL timer classes
 */

#ifndef KWIVER_VITAL_WALL_TIMER_H
#define KWIVER_VITAL_WALL_TIMER_H

#include <vital/util/timer.h>

#if VITAL_USE_CHRONO

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
class VITAL_EXPORT wall_timer
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
    m_start = std::chrono::system_clock::now();
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
    m_end = std::chrono::system_clock::now();
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
      std::chrono::duration< double > elapsed_seconds = std::chrono::system_clock::now() - m_start;
      return elapsed_seconds.count();
    }
    else
    {
      std::chrono::duration< double > elapsed_seconds = m_end - m_start;
      return elapsed_seconds.count();
    }
  }

private:

  std::chrono::time_point< std::chrono::system_clock > m_start;
  std::chrono::time_point< std::chrono::system_clock > m_end;

}; // end class wall_timer

template class VITAL_EXPORT scoped_timer< wall_timer >;
typedef scoped_timer< wall_timer > scoped_wall_timer;

} }   // end namespace

#else // ==================================================================

namespace kwiver {
namespace vital {

/*
 * Empty implementation where chrono is not supported.
 */
class VITAL_EXPORT wall_timer
  : public timer
{
public:
  wall_timer()  { }
  virtual ~wall_timer() { }

  virtual bool timer_available() { return false; }

  virtual void start() { }
  virtual void stop() { }
  virtual double elapsed() const { return 0; }
}; // end class wall_timer

template class scoped_timer< wall_timer >;
typedef scoped_timer< wall_timer > scoped_wall_timer;

} } // end namespace

#endif

#endif /* KWIVER_VITAL_SCOPED_TIMER_H */
