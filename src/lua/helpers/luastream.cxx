/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "luastream.h"

#include <luabind/class.hpp>

#include <algorithm>
#include <string>

#include <cstddef>

luaistream_device
::luaistream_device(luabind::object const& obj)
  : m_obj(obj)
{
}

luaistream_device
::~luaistream_device()
{
}

std::streamsize
luaistream_device
::read(char_type* s, std::streamsize n)
{
  luabind::object bytes = luabind::call_member<luabind::object>(m_obj, "read", n);

  long const sz = luabind::call_member<long>(m_obj, "__len", bytes);

  if (sz)
  {
    std::string cppstr = luabind::object_cast<std::string>(bytes);

    std::copy(cppstr.begin(), cppstr.end(), s);

    return sz;
  }
  else
  {
    return -1;
  }
}

luaostream_device
::luaostream_device(luabind::object const& obj)
  : m_obj(obj)
{
}

luaostream_device
::~luaostream_device()
{
}

std::streamsize
luaostream_device
::write(char_type const* s, std::streamsize n)
{
  std::string const bytes(s, n);

  luabind::call_member<luabind::object>(m_obj, "write", bytes);

  return n;
}
