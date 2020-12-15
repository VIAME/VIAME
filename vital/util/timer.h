// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface and implementation of timer classes
 */

#ifndef KWIVER_VITAL_TIMER_H
#define KWIVER_VITAL_TIMER_H

#include <vital/vital_config.h>
#include <vital/vital_export.h>

#include <iostream>
#include <string>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Abstract base class for timers.
 *
 * This class represents an interval timer.
 */
class timer
{
public:
  timer()
    : m_active(false)
  { }

  virtual ~timer()
  {
    m_active = false;
  }

  /**
   * @brief Is timer available.
   *
   * This method is used to determine if the timer has a real
   * implementation. In some cases, the current system does not have
   * the support for certain types of timers. If there is no support,
   * then this method returns false.
   *
   * If a timer is not supported, the API is available but the timer
   * doesn't time.
   *
   * @return \b true if the timer is supported, \b false if not.
   */
  virtual bool timer_available() { return false; }

  /**
   * @brief Start the timer.
   *
   * The clock time when this timer is started is saved.
   */
  virtual void start() = 0;

  /**
   * @brief Stop the timer.
   *
   * The time this clock was stopped is saved. This value is used to
   * determine the elapsed time.
   */
  virtual void stop() = 0;

  /**
   * @brief Calculate elapsed time.
   *
   * The elapsed time of this timer is returned. This method works if
   * the timer is stopped or still running.
   *
   * @return Seconds since the timer was started.
   */
  virtual double elapsed() const = 0;

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

protected:

  bool m_active;

}; // end class timer

// -----------------------------------------------------------------
/**
 * @brief Scoped timer.
 *
 * This class represents a timer that measures the lifetime of
 * a timer class in seconds.
 */
template < class timer_t >
class scoped_timer
{
public:
  scoped_timer()
  {
    m_timer.start();
  }

  scoped_timer( std::string const& title )
    : m_title( title )
  {
    m_timer.start();
  }

  ~scoped_timer()
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

    std::cerr << "elapsed time: " << interval << " sec\n";
  }

  std::string m_title; //< optional measurement title string

private:

  timer_t m_timer;

}; // end class scoped timer

} } // end namespace

#endif /* KWIVER_VITAL_TIMER_H */
