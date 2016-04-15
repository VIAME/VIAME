/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
