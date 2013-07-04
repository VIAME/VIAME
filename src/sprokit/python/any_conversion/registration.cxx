/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include <sprokit/python/util/python_gil.h>

#include <boost/python/converter/registry.hpp>
#include <boost/python/errors.hpp>
#include <boost/python/to_python_converter.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/once.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/any.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>

#include <map>

/**
 * \file any_conversion/registration.cxx
 *
 * \brief Helpers for working with boost::any in Python.
 */

using namespace boost::python;

namespace sprokit
{

namespace python
{

namespace
{

class any_converter
{
  public:
    static void* convertible(PyObject* obj);
    static PyObject* convert(boost::any const& any);
    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data);

    static void add_from(priority_t priority, from_any_func_t from);
    static void add_to(priority_t priority, to_any_func_t to);
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

static void register_to_python();

void
register_conversion(priority_t priority, from_any_func_t from, to_any_func_t to)
{
  static boost::once_flag once;

  boost::call_once(once, register_to_python);

  if (from)
  {
    any_converter::add_from(priority, from);
  }

  if (to)
  {
    any_converter::add_to(priority, to);
  }
}

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

void*
any_converter
::convertible(PyObject* obj)
{
  return obj;
}

PyObject*
any_converter
::convert(boost::any const& any)
{
  python_gil const gil;

  (void)gil;

  if (any.empty())
  {
    Py_RETURN_NONE;
  }

  boost::python::type_info const info(any.type());

  converter::registration const* const reg = converter::registry::query(info);

  if (reg)
  {
    boost::shared_lock<boost::shared_mutex> const lock(m_mutex);

    (void)lock;

    BOOST_FOREACH (to_map_t::value_type const& to, m_to)
    {
      to_any_func_t const& func = to.second;

      try
      {
        opt_pyobject_t const opt = func(any);

        if (opt)
        {
          return *opt;
        }
      }
      catch (error_already_set const&)
      {
      }
    }
  }

  /// \todo Log that the any has a type which is not supported yet.

  Py_RETURN_NONE;
}

void
any_converter
::construct(PyObject* obj, converter::rvalue_from_python_stage1_data* data)
{
  python_gil const gil;

  (void)gil;

  void* storage = reinterpret_cast<converter::rvalue_from_python_storage<boost::any>*>(data)->storage.bytes;

  if (obj != Py_None)
  {
    boost::shared_lock<boost::shared_mutex> const lock(m_mutex);

    (void)lock;

    BOOST_FOREACH (from_map_t::value_type const& from, m_from)
    {
      from_any_func_t const& func = from.second;

      if (func(obj, storage))
      {
        data->convertible = storage;
        return;
      }
    }
  }

  new (storage) boost::any;
  data->convertible = storage;
}

}

void
register_to_python()
{
  python_gil const gil;

  (void)gil;

  to_python_converter<boost::any, any_converter>();
  converter::registry::push_back(
    &any_converter::convertible,
    &any_converter::construct,
    type_id<boost::any>());
}

}

}
