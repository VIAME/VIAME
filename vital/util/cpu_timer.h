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

#include <vital/vital_config.h>

#if VITAL_USE_CHRONO

#include <string>
#include <chrono>
#include <iostream>


namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Interval timer class.
 *
 * This class represents an interval timer.
 */
class cpu_timer
{
public:
  cpu_timer()
    : m_active(false)
  { }


  ~cpu_timer()
  {
    m_active = false;
  }

  static bool timer_available() { return true; }

  /**
   * @brief Start the timer.
   *
   * The clock time when this timer is started is saved.
   */
  void start()
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
  void stop()
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
  double elapsed() const
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


  /**
   * @brief Is timer active
   *
   * This method returns whether the timer is active or not.
   *
   * @return \b true if timer is currently active, \b false if not.
   */
  bool is_active() const
  {
    return m_active;
  }


private:

  std::chrono::time_point< std::chrono::system_clock > m_start;
  std::chrono::time_point< std::chrono::system_clock > m_end;
  bool m_active;

}; // end class cpu_timer



// -----------------------------------------------------------------
/**
 * @brief Scoped CPU timer.
 *
 * This class represents a CPU timer that measures the lifetime of
 * this class in seconds.
 */
class scoped_cpu_timer
{
public:
  scoped_cpu_timer()
  {
    m_timer.start();
  }

  scoped_cpu_timer( std::string const& title )
    : m_title( title )
  {
    m_timer.start();
  }

  ~scoped_cpu_timer()
  {
    m_timer.stop();
    format_interval( m_timer.elapsed() );
  }

protected:
  /**
   * @brief Format time interval.
   *
   * This method formats the elapsed time interval. Derived classes
   * can redirect the output to a logger or a file as desired.
   *
   * @param interval Number of seconds in interval
   */
  void format_interval( double interval )
  {
    if ( ! m_title.empty() )
    {
      std::cerr << m_title << " - ";
    }

    std::cerr << "elapsed time: " << interval << "sec\n";
  }

  std::string m_title; //< optional measurement title string

private:

  cpu_timer m_timer;

}; // end class cpu_timer

} }   // end namespace

#else // ==================================================================

namespace kwiver {
namespace vital {

/*
 * Empty implementation where chrono is not supported.
 */
class cpu_timer
{
public:
  cpu_timer()  { }
  ~cpu_timer() { }

  static bool timer_available() { return false; }

  void start() { }
  void stop() { }
  double elapsed() const { return 0; }
  bool is_active() const { return false; }
}; // end class cpu_timer


// ------------------------------------------------------------------
class scoped_cpu_timer
{
public:
  // -- CONSTRUCTORS --
  scoped_cpu_timer() { }
  scoped_cpu_timer( std::string const& ) { }

  ~scoped_cpu_timer() { }

}; // end class scoped_timer

} }   // end namespace

#endif

#endif /* KWIVER_VITAL_SCOPED_TIMER_H */
