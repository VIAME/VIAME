/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "timestamp.h"

#include <cmath>

/**
 * \file timestamp.cxx
 *
 * \brief Implementation of the timestamp class.
 */

namespace vistk
{

timestamp
::timestamp()
  : m_time()
  , m_frame()
{
}

timestamp
::timestamp(time_t t)
  : m_time(t)
  , m_frame()
{
}

timestamp
::timestamp(frame_t f)
  : m_time()
  , m_frame(f)
{
}

timestamp
::timestamp(time_t t, frame_t f)
  : m_time(t)
  , m_frame(f)
{
}

timestamp
::~timestamp()
{
}

bool
timestamp
::has_time() const
{
  return m_time;
}

bool
timestamp
::has_frame() const
{
  return m_frame;
}

timestamp::time_t
timestamp
::time() const
{
  return *m_time;
}

timestamp::frame_t
timestamp
::frame() const
{
  return *m_frame;
}

bool
timestamp
::is_valid() const
{
  return (has_time() || has_frame());
}

static bool double_compare(double a, double b);

bool
timestamp
::operator == (timestamp const& ts) const
{
  if (has_time() && ts.has_time())
  {
    return double_compare(time(), ts.time());
  }
  else if (has_time() || ts.has_time())
  {
    /// \todo Log a warning.
    return false;
  }

  if (has_frame() && ts.has_frame())
  {
    return (frame() == ts.frame());
  }
  else if (has_frame() || ts.has_frame())
  {
    /// \todo Log a warning.
    return false;
  }

  /// \todo Log a warning.
  return false;
}

bool
timestamp
::operator <  (timestamp const& ts) const
{
  if (has_time() && ts.has_time())
  {
    return (time() < ts.time());
  }
  else if (has_time() || ts.has_time())
  {
    /// \todo Log a warning.
    return false;
  }

  if (has_frame() && ts.has_frame())
  {
    return (frame() < ts.frame());
  }
  else if (has_frame() || ts.has_frame())
  {
    /// \todo Log a warning.
    return false;
  }

  /// \todo Log a warning.
  return false;
}

bool
double_compare(double a, double b)
{
  static double const epsilon = 0.00001;

  double const diff = a - b;
  double const abs_diff = fabs(diff);

  return (epsilon > abs_diff);
}

}
