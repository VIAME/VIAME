/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_HELPERS_PYTHON_CONVERT_OPTIONAL_H
#define VISTK_PYTHON_HELPERS_PYTHON_CONVERT_OPTIONAL_H

#include <boost/python/converter/registry.hpp>
#include <boost/python/class.hpp>
#include <boost/python/default_call_policies.hpp>
#include <boost/python/implicit.hpp>
#include <boost/optional.hpp>

#include <Python.h>

/**
 * \file python_convert_optional.h
 *
 * \brief Helpers for working with boost::optional in Python.
 */

template <typename T>
class boost_optional_converter
{
  public:
    typedef T type_t;
    typedef boost::optional<T> optional_t;

    boost_optional_converter()
    {
      boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        boost::python::type_id<optional_t>());
    }
    ~boost_optional_converter()
    {
    }

    static void* convertible(PyObject* obj)
    {
      if (obj == Py_None)
      {
        return obj;
      }

      using namespace boost::python::converter;

      registration const& converters(registered<type_t>::converters);

      if (implicit_rvalue_convertible_from_python(obj, converters))
      {
        rvalue_from_python_stage1_data data = rvalue_from_python_stage1(obj, converters);
        return rvalue_from_python_stage2(obj, data, converters);
      }

      return NULL;
    }

    static PyObject* convert(optional_t const& opt)
    {
      if (opt)
      {
        return boost::python::to_python_value<T>()(*opt);
      }
      else
      {
        Py_RETURN_NONE;
      }
    }

    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
      void* storage = reinterpret_cast<boost::python::converter::rvalue_from_python_storage<optional_t>*>(data)->storage.bytes;

#define CONSTRUCT(args)            \
  do                               \
  {                                \
    new (storage) optional_t args; \
    data->convertible = storage;   \
    return;                        \
  } while (false)

      if (obj == Py_None)
      {
        CONSTRUCT(());
      }
      else
      {
        type_t* t = reinterpret_cast<type_t*>(data->convertible);
        CONSTRUCT((*t));
      }

#undef CONSTRUCT
    }
};

template <typename T>
void register_optional_converter(char const* name, char const* desc);

template <typename T>
class boost_optional_operations
{
  public:
    typedef T type_t;
    typedef boost::optional<T> optional_t;

    static bool not_(optional_t const& opt);
    static type_t& get(optional_t& opt);
    static type_t const& get_default(optional_t const& opt, type_t const& def);
};

template <typename T>
void
register_optional_converter(char const* name, char const* desc)
{
  typedef boost_optional_converter<T> converter_t;
  typedef typename converter_t::optional_t optional_t;

  boost::python::class_<optional_t>(name
    , desc
    , boost::python::no_init)
    .def(boost::python::init<>())
    .def(boost::python::init<T>())
    .def("empty", &boost_optional_operations<T>::not_
      , "True if there is no value, False otherwise.")
    .def("get", &boost_optional_operations<T>::get
      , "Returns the contained value, or None if there isn\'t one."
      , boost::python::return_internal_reference<>())
    .def("get", &boost_optional_operations<T>::get_default
      , (boost::python::arg("default"))
      , "Returns the contained value, or default if there isn\'t one."
      , boost::python::return_internal_reference<>())
  ;

  boost::python::to_python_converter<optional_t, converter_t>();
  converter_t();

  boost::python::implicitly_convertible<T, optional_t>();
}

template <typename T>
bool
boost_optional_operations<T>
::not_(optional_t const& opt)
{
  return !opt;
}

template <typename T>
T&
boost_optional_operations<T>
::get(optional_t& opt)
{
  return opt.get();
}

template <typename T>
T const&
boost_optional_operations<T>
::get_default(optional_t const& opt, T const& def)
{
  return opt.get_value_or(def);
}

#endif // VISTK_PYTHON_PIPELINE_PYTHON_CONVERT_OPTIONAL_H
