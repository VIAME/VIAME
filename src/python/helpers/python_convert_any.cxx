/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "python_convert_any.h"

#include <boost/python/converter/registry.hpp>
#include <boost/python/extract.hpp>
#include <boost/cstdint.hpp>

/**
 * \file python_convert_any.cxx
 *
 * \brief Helpers for working with boost::any in Python.
 */

using namespace boost::python;

boost_any_to_object
::boost_any_to_object()
{
  converter::registry::push_back(
    &convertible,
    &construct,
    type_id<boost::any>());
}

boost_any_to_object
::~boost_any_to_object()
{
}

void*
boost_any_to_object
::convertible(PyObject* obj)
{
  return obj;
}

#define REGISTER_TYPES(call) \
  call(std::string);         \
  call(bool);                \
  call(char);                \
  call(char signed);         \
  call(char unsigned);       \
  call(short signed);        \
  call(short unsigned);      \
  call(int signed);          \
  call(int unsigned);        \
  call(long signed);         \
  call(long unsigned);       \
  call(float);               \
  call(double);              \
  call(long double)

PyObject*
boost_any_to_object
::convert(boost::any const& a)
{
  if (a.empty())
  {
    return detail::none();
  }

  type_info const info(a.type());

#define TRY_CONVERT_TO(T)                      \
  do                                           \
  {                                            \
    converter::registration const* const reg = \
      converter::registry::query(info);        \
    if (reg)                                   \
    {                                          \
      try                                      \
      {                                        \
        T const t = boost::any_cast<T>(a);     \
        object const o(t);                     \
        return incref(o.ptr());                \
      }                                        \
      catch (boost::bad_any_cast&)             \
      {                                        \
      }                                        \
      catch (error_already_set&)               \
      {                                        \
        /** \todo Log that there is not a */   \
        /** known converter for the type. */   \
        return detail::none();                 \
      }                                        \
    }                                          \
  } while (false)

  REGISTER_TYPES(TRY_CONVERT_TO);

#undef TRY_CONVERT_TO

  /// \todo Log that the any has a type which is not supported yet.

  return detail::none();
}

void
boost_any_to_object
::construct(PyObject* obj, converter::rvalue_from_python_stage1_data* data)
{
  void* storage = reinterpret_cast<converter::rvalue_from_python_storage<boost::any>*>(data)->storage.bytes;

#define CONSTRUCT(args)            \
  do                               \
  {                                \
    new (storage) boost::any args; \
    data->convertible = storage;   \
    return;                        \
  } while (false)

  if (obj == Py_None)
  {
    CONSTRUCT(());
  }

#define TRY_CONVERT_FROM_RAW(T)        \
  do                                   \
  {                                    \
    extract<T> ex(obj);                \
    if (ex.check())                    \
    {                                  \
      CONSTRUCT((ex()));               \
    }                                  \
  } while (false)
#define TRY_CONVERT_FROM(T)      \
  TRY_CONVERT_FROM_RAW(T);       \
  TRY_CONVERT_FROM_RAW(T const); \
  TRY_CONVERT_FROM_RAW(T const&)

  REGISTER_TYPES(TRY_CONVERT_FROM);

#undef TRY_CONVERT_FROM
#undef TRY_CONVERT_FROM_RAW

  CONSTRUCT(());

#undef CONSTRUCT
}

#undef REGISTER_TYPES
