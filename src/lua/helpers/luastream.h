/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_LUA_HELPERS_LUASTREAM_H
#define VISTK_LUA_HELPERS_LUASTREAM_H

extern "C"
{
#include <lua.h>
}

#include <luabind/object.hpp>

#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/stream.hpp>

#include <iostream>

class luaistream_device
  : public boost::iostreams::source
{
  public:
    luaistream_device(luabind::object const& obj);
    ~luaistream_device();

    std::streamsize read(char_type* s, std::streamsize n);
  private:
    luabind::object m_obj;
};

typedef boost::iostreams::stream<luaistream_device> luaistream;

class luaostream_device
  : public boost::iostreams::sink
{
  public:
    luaostream_device(luabind::object const& obj);
    ~luaostream_device();

    std::streamsize write(char_type const* s, std::streamsize n);
  private:
    luabind::object m_obj;
};

typedef boost::iostreams::stream<luaostream_device> luaostream;

#endif // VISTK_LUA_HELPERS_LUASTREAM_H
