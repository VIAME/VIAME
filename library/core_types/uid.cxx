// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of viame_algorithm_framework global uid

#include "uid.h"

namespace viame {

// ----------------------------------------------------------------------------
uid
::uid( const std::string& data )
  : m_uid( data )
{}

uid
::uid( const char* data, size_t byte_count )
  : m_uid( data, byte_count )
{}

uid
::uid()
{}

// ----------------------------------------------------------------------------
bool
uid
::is_valid() const
{
  return !m_uid.empty();
}

// ----------------------------------------------------------------------------
std::string const&
uid
::value() const
{
  return m_uid;
}

// ----------------------------------------------------------------------------
size_t
uid
::size() const
{
  return m_uid.size();
}

// ----------------------------------------------------------------------------
bool
uid::operator==( const uid& other ) const
{
  return this->m_uid == other.m_uid;
}

// ----------------------------------------------------------------------------
bool
uid::operator!=( const uid& other ) const
{
  return this->m_uid != other.m_uid;
}

// ----------------------------------------------------------------------------
bool
uid::operator<( const uid& other ) const
{
  return this->m_uid < other.m_uid;
}

} // namespace viame
