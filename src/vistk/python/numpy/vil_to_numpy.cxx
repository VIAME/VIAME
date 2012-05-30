/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "vil_to_numpy.h"

#include "registration.h"
#include "type_mappings.h"

#include <vistk/python/util/python_gil.h>

#include <boost/python/object.hpp>

#include <vil/vil_image_view.h>
#include <vil/vil_image_view_base.h>

#include <Python.h>

#include <numpy/arrayobject.h>

/**
 * \file vil_to_numpy.cxx
 *
 * \brief Implementation of a vil-to-NumPy converter function.
 */

using namespace boost::python;

namespace vistk
{

namespace python
{

PyObject*
vil_to_numpy(vil_image_view_base_sptr const& img)
{
  vistk::python::python_gil const gil;

  (void) gil;

  register_image_base();

  vil_pixel_format const vil_format = img->pixel_format();

  if (!vil_pixel_format_sizeof_components(vil_format))
  {
    Py_RETURN_NONE;
  }

  switch (vil_format)
  {
#define CONVERT_FORMAT(vil_type)                                            \
  case vil_type:                                                            \
  {                                                                         \
    typedef vil_pixel_format_type_of<vil_type>::type pixel_t;               \
    typedef vil_pixel_format_type_of<vil_type>::component_type component_t; \
                                                                            \
    vil_image_view<component_t> const i = img;                              \
                                                                            \
    return vil_to_numpy<component_t>(i);                                    \
  }

#if VXL_HAS_INT_64
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_UINT_64)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_INT_64)
#endif
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_UINT_32)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_INT_32)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_UINT_16)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_INT_16)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_BYTE)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_SBYTE)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_FLOAT)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_DOUBLE)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_BOOL)

#if VXL_HAS_INT_64
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_UINT_64)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_INT_64)
#endif
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_UINT_32)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_INT_32)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_UINT_16)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_INT_16)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_BYTE)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_SBYTE)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_FLOAT)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGB_DOUBLE)

#if VXL_HAS_INT_64
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_UINT_64)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_INT_64)
#endif
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_UINT_32)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_INT_32)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_UINT_16)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_INT_16)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_BYTE)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_SBYTE)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_FLOAT)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_RGBA_DOUBLE)

/// \todo Is there a define for C++11?
#if 0
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_COMPLEX_FLOAT)
    CONVERT_FORMAT(VIL_PIXEL_FORMAT_COMPLEX_DOUBLE)
#else
    case VIL_PIXEL_FORMAT_COMPLEX_FLOAT:
    case VIL_PIXEL_FORMAT_COMPLEX_DOUBLE:
#endif

#undef CONVERT_FORMAT

    case VIL_PIXEL_FORMAT_UNKNOWN:
    case VIL_PIXEL_FORMAT_ENUM_END:
    default:
      break;
  }

  Py_RETURN_NONE;
}

template <typename T>
static NPY_TYPES python_pixel_type();

template <typename T>
PyObject*
vil_to_numpy(vil_image_view<T> const& img)
{
  vistk::python::python_gil const gil;

  (void) gil;

  typedef vil_image_view<T> image_t;
  typedef typename image_t::pixel_type pixel_t;

  register_memory_chunk();
  register_image_type<pixel_t>();

  NPY_TYPES const arr_type = python_pixel_type<pixel_t>();
  int const nd = 3;
  npy_intp* const dims = PyDimMem_NEW(nd);
  npy_intp* const strides = PyDimMem_NEW(nd);
  bool const contig = img.is_contiguous();

  dims[0] = img.ni();
  dims[1] = img.nj();
  dims[2] = img.nplanes();

  strides[0] = img.istep();
  strides[1] = img.jstep();
  strides[2] = img.planestep();

  int flags = 0;

  if (contig)
  {
    if (img.planestep() == 1)
    {
      flags |= NPY_CONTIGUOUS;
    }
    else if (img.istep() == 1)
    {
      flags |= NPY_FORTRAN;
    }
  }

  flags |= NPY_WRITEABLE;
  flags |= NPY_NOTSWAPPED;

  uintptr_t const mem = reinterpret_cast<uintptr_t>(img.top_left_ptr());

  if (!(mem % sizeof(T)))
  {
    flags |= NPY_ALIGNED;
  }

  PyObject* obj = Py_None;

  if (img.memory_chunk())
  {
    object bp_obj = object(img.memory_chunk());
    obj = bp_obj.ptr();
  }
  else
  {
    /// \todo Log that vil doesn't own this memory...there be dragons here.
    Py_INCREF(obj);
  }

  PyObject* const arr = PyArray_New(&PyArray_Type, nd, dims, arr_type, strides, reinterpret_cast<void*>(mem), sizeof(pixel_t), flags, obj);

  PyDimMem_FREE(dims);
  PyDimMem_FREE(strides);

  return arr;
}

#define CONVERT_FORMAT(npy_type, cpp_type) \
  template <>                              \
  NPY_TYPES                                \
  python_pixel_type<cpp_type>()            \
  {                                        \
    return npy_type;                       \
  }

FORMAT_CONVERSION(CONVERT_FORMAT, NONE)

#undef PIXEL_CONVERSION

template <typename T>
NPY_TYPES
python_pixel_type()
{
  return NPY_USERDEF;
}

}

}
