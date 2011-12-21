/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include <boost/thread/locks.hpp>
#include <boost/thread/once.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/any.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

#include <map>

/**
 * \file any_conversion/registration.cxx
 *
 * \brief Helpers for working with boost::any in Lua.
 */

namespace vistk
{

namespace lua
{

namespace
{

class any_converter;
typedef boost::shared_ptr<any_converter> any_converter_t;

class any_converter
{
  public:
    static void add_from(priority_t priority, from_any_func_t from);
    static void add_to(priority_t priority, to_any_func_t to);

    static opt_any_t from(lua_State* L, int index);
    static bool to(lua_State* L, boost::any const& any);
  private:
    static boost::shared_mutex m_mutex;

    typedef std::multimap<priority_t, from_any_func_t> from_map_t;
    typedef std::multimap<priority_t, to_any_func_t> to_map_t;

    static from_map_t m_from;
    static to_map_t m_to;
};

boost::shared_mutex any_converter::m_mutex;

any_converter::from_map_t any_converter::m_from = any_converter::from_map_t();
any_converter::to_map_t any_converter::m_to = any_converter::to_map_t();

}

void
register_conversion(priority_t priority, from_any_func_t from, to_any_func_t to)
{
  if (from)
  {
    any_converter::add_from(priority, from);
  }

  if (to)
  {
    any_converter::add_to(priority, to);
  }
}

}

}

namespace luabind
{

int
default_converter<boost::any>
::compute_score(lua_State* /*L*/, int /*index*/)
{
  return 0;
}

boost::any
default_converter<boost::any>
::from(lua_State* L, int index)
{
  vistk::lua::opt_any_t const oany = vistk::lua::any_converter::from(L, index);

  if (oany)
  {
    return oany;
  }

  /// \todo Log a warning that the requested type has not been registered.

  return boost::any();
}

void
default_converter<boost::any>
::to(lua_State* L, boost::any const& any)
{
  if (any.empty())
  {
    lua_pushnil(L);
    return;
  }

  if (!vistk::lua::any_converter::to(L, any))
  {
    /// \todo Log that the any has a type which is not supported yet.

    lua_pushnil(L);
  }
}

}

namespace vistk
{

namespace lua
{

namespace
{

void
any_converter
::add_from(priority_t priority, from_any_func_t from)
{
  boost::unique_lock<boost::shared_mutex> const lock(m_mutex);

  (void)lock;

  m_from.insert(from_map_t::value_type(priority, from));
}

void
any_converter
::add_to(priority_t priority, to_any_func_t to)
{
  boost::unique_lock<boost::shared_mutex> const lock(m_mutex);

  (void)lock;

  m_to.insert(to_map_t::value_type(priority, to));
}

opt_any_t
any_converter
::from(lua_State* L, int index)
{
  boost::shared_lock<boost::shared_mutex> const lock(m_mutex);

  (void)lock;

  BOOST_FOREACH (from_map_t::value_type const& from, m_from)
  {
    opt_any_t const oany = from.second(L, index);

    if (oany)
    {
      return oany;
    }
  }

  return opt_any_t();
}

bool
any_converter
::to(lua_State* L, boost::any const& any)
{
  boost::shared_lock<boost::shared_mutex> const lock(m_mutex);

  (void)lock;

  BOOST_FOREACH (to_map_t::value_type const& to, m_to)
  {
    if (to.second(L, any))
    {
      return true;
    }
  }

  return false;
}

}

}

}
