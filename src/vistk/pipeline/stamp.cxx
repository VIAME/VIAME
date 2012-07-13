/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "stamp.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

/**
 * \file stamp.cxx
 *
 * \brief Implementation of \link vistk::stamp stamps\endlink.
 */

namespace vistk
{

stamp::color_t stamp::m_new_color = color_t(0);

stamp_t
stamp
::new_stamp()
{
  static boost::mutex mut;

  stamp_t st;

  {
    boost::mutex::scoped_lock const lock(mut);

    (void)lock;

    st = stamp_t(new stamp(m_new_color, 0));
    ++m_new_color;
  }

  return st;
}

stamp_t
stamp
::incremented_stamp(stamp_t const& st)
{
  /// \todo Check \p st for \c NULL?

  return stamp_t(new stamp(st->m_color, st->m_index + 1));
}

stamp_t
stamp
::recolored_stamp(stamp_t const& st, stamp_t const& st2)
{
  /// \todo Check \p st for \c NULL?

  return stamp_t(new stamp(st2->m_color, st->m_index));
}

bool
stamp
::is_same_color(stamp_t const& st) const
{
  /// \todo Check \p st for \c NULL?

  return (m_color == st->m_color);
}

bool
stamp
::operator == (stamp const& st) const
{
  /// \todo Check \p st for \c NULL?

  return ((m_color == st.m_color) && (m_index == st.m_index));
}

bool
stamp
::operator <  (stamp const& st) const
{
  /// \todo Check \p st for \c NULL?

  return ((m_color == st.m_color) && (m_index < st.m_index));
}

stamp
::stamp(color_t color, index_t index)
  : m_color(color)
  , m_index(index)
{
}

}
