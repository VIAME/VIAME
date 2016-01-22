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

#ifndef KWIVER_VITAL_SCOPED_TIMER_H
#define KWIVER_VITAL_SCOPED_TIMER_H

#include <vital/vital_config.h>

#if VITAL_USE_CHRONO

#include <string>
#include <chrono>
#include <iostream>


namespace kwiver {
namespace vital {

// -----------------------------------------------------------------
/**
 * @brief Scoped CPU timer.
 *
 * This class represents a scoped timer that measures the lifetime of
 * this class in seconds.
 */
class scoped_timer
{
public:
  scoped_timer()
  {
    start();
  }

  scoped_timer( std::string const& title )
    : m_title( title )
  {
    start();
  }

  ~scoped_timer()
  {
    stop();
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
  void start()
  {
    m_start = std::chrono::system_clock::now();
  }


  void stop()
  {
    m_end = std::chrono::system_clock::now();
    std::chrono::duration< double > elapsed_seconds = m_end - m_start;
    format_interval( elapsed_seconds.count() );
  }


  std::chrono::time_point< std::chrono::system_clock > m_start;
  std::chrono::time_point< std::chrono::system_clock > m_end;

}; // end class scoped_timer

} }   // end namespace

#else

namespace kwiver {
namespace vital {

// -----------------------------------------------------------------
/*
 * Empty implementation where chrono is not supported.
 */
class scoped_timer
{
public:
  // -- CONSTRUCTORS --
  scoped_timer() { }
  scoped_timer( std::string const& ) { }

  ~scoped_timer() { }

}; // end class scoped_timer

#endif

#endif /* KWIVER_VITAL_SCOPED_TIMER_H */
