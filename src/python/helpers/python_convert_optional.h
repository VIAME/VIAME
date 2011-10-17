/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_HELPERS_PYTHON_CONVERT_OPTIONAL_H
#define VISTK_PYTHON_HELPERS_PYTHON_CONVERT_OPTIONAL_H

#include <boost/optional.hpp>
#include <boost/python/converter/registry.hpp>

#include <Python.h>

/**
 * \file python_convert_any.h
 *
 * \brief Helpers for working with boost::any in Python.
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
        return boost::python::detail::none();
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

#define REGISTER_OPTIONAL_CONVERTER(T)             \
  do                                               \
  {                                                \
    typedef boost_optional_converter<T> converter; \
    typedef typename converter::optional_t opt_t;  \
    to_python_converter<opt_t, converter>();       \
    converter();                                   \
  } while (false)

#endif // VISTK_PYTHON_PIPELINE_PYTHON_CONVERT_OPTIONAL_H
