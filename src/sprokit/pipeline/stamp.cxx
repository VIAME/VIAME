/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "stamp.h"

/**
 * \file stamp.cxx
 *
 * \brief Implementation of \link sprokit::stamp stamps\endlink.
 */

namespace sprokit
{

stamp_t
stamp
::new_stamp(increment_t increment)
{
  return stamp_t(new stamp(increment, 0));
}

stamp_t
stamp
::incremented_stamp(stamp_t const& st)
{
  /// \todo Check \p st for \c NULL?

  return stamp_t(new stamp(st->m_increment, st->m_index + st->m_increment));
}

bool
stamp
::operator == (stamp const& st) const
{
  /// \todo Check \p st for \c NULL?

  return (m_index == st.m_index);
}

bool
stamp
::operator <  (stamp const& st) const
{
  /// \todo Check \p st for \c NULL?

  return (m_index < st.m_index);
}

stamp
::stamp(increment_t increment, index_t index)
  : m_increment(increment)
  , m_index(index)
{
}

}
