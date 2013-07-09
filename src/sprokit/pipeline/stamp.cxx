/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "stamp.h"

#include <stdexcept>

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
  if (!st)
  {
    static const std::string reason = "A NULL stamp cannot be incremented";

    throw std::runtime_error(reason);
  }

  return stamp_t(new stamp(st->m_increment, st->m_index + st->m_increment));
}

bool
stamp
::operator == (stamp const& st) const
{
  return (m_index == st.m_index);
}

bool
stamp
::operator <  (stamp const& st) const
{
  return (m_index < st.m_index);
}

stamp
::stamp(increment_t increment, index_t index)
  : m_increment(increment)
  , m_index(index)
{
}

}
