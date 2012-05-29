/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "numpy_to_vil.h"

#include "numpy_memory_chunk.h"
#include "registration.h"
#include "type_mappings.h"

#include <vistk/python/util/python_gil.h>

#include <boost/python/extract.hpp>

#include <vil/vil_image_view.h>
#include <vil/vil_image_view_base.h>

#include <stdexcept>

#include <Python.h>

#include <numpy/arrayobject.h>

using namespace boost::python;

namespace vistk
{

namespace python
{

static void numpy_to_vil_check(PyObject* obj);
template <typename T>
static vil_image_view_base_sptr convert_image(PyArrayObject* arr);

vil_image_view_base_sptr
numpy_to_vil(PyObject* obj)
{
  vistk::python::python_gil gil;

  (void)gil;

  numpy_to_vil_check(obj);

  register_image_base();

  PyArrayObject* const arr = reinterpret_cast<PyArrayObject*>(obj);
  int const type = PyArray_TYPE(arr);

  switch (type)
  {

#define CONVERT_IMAGE(npy_type, cpp_type) \
  case npy_type:                          \
    return convert_image<cpp_type>(arr)

    FORMAT_CONVERSION(CONVERT_IMAGE, LINES)

#undef CONVERT_IMAGE

    default:
      break;
  }

  return vil_image_view_base_sptr();
}

void
numpy_to_vil_check(PyObject* obj)
{
  if (obj == Py_None)
  {
    static std::string const reason = "Unable to convert a None object";

    throw std::runtime_error(reason);
  }

  if (!PyArray_Check(obj))
  {
    static std::string const reason = "Object given was not a NumPy array";

    throw std::runtime_error(reason);
  }

  PyArrayObject* const arr = reinterpret_cast<PyArrayObject*>(obj);
  int const nd = PyArray_NDIM(arr);

  if ((nd != 2) && (nd != 3))
  {
    static std::string const reason = "Array does not have 2 or 3 dimensions";

    throw std::runtime_error(reason);
  }
}

template <typename T>
vil_image_view_base_sptr
convert_image(PyArrayObject* arr)
{
  typedef vil_image_view<T> image_t;
  typedef typename image_t::pixel_type pixel_t;

  register_memory_chunk();
  register_image_type<pixel_t>();

  int const nd = PyArray_NDIM(arr);
  npy_intp const* const dims = PyArray_DIMS(arr);
  npy_intp const* const strides = PyArray_STRIDES(arr);
  void* const mem = PyArray_DATA(arr);
  size_t const pxsz = PyArray_ITEMSIZE(arr);
  int const flags = PyArray_FLAGS(arr);

  vil_memory_chunk_sptr chunk;

  if (~flags & NPY_UPDATEIFCOPY)
  {
    PyObject* const memobj = PyArray_BASE(arr);

    if (memobj)
    {
      extract<vil_memory_chunk&> ex(memobj);

      if (ex.check())
      {
        chunk = &ex();
      }
    }
  }

  if (!chunk)
  {
    chunk = new numpy_memory_chunk(arr);
  }

  size_t const np = ((nd == 2) ? 1 : dims[2]);
  ptrdiff_t const pstep = ((nd == 2) ? 0 : strides[2]);

  vil_image_view_base_sptr const base = new image_t(chunk, reinterpret_cast<pixel_t*>(mem),
                                                    dims[0], dims[1], np,
                                                    strides[0] / pxsz, strides[1] / pxsz, pstep / pxsz);

  return base;
}

}

}
