/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/numpy_memory_chunk.h>
#include <python/helpers/python_wrap_vil_smart_ptr.h>

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>

#include <boost/python/converter/registry.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/object.hpp>
#include <boost/cstdint.hpp>

#include <vil/vil_image_view.h>
#include <vil/vil_memory_chunk.h>

#include <Python.h>

#include <numpy/arrayobject.h>

/**
 * \file vil.cxx
 *
 * \brief Python bindings for \link vil_image_view\endlink.
 */

using namespace boost::python;

template <typename T>
static NPY_TYPES python_pixel_type();

template <typename T>
class vil_image_converter
{
  public:
    typedef T image_t;
    typedef typename image_t::pixel_type pixel_t;

    vil_image_converter()
    {
      boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        boost::python::type_id<image_t>());
    }
    ~vil_image_converter()
    {
    }

    static void* convertible(PyObject* obj)
    {
      if (obj == Py_None)
      {
        return NULL;
      }

      if (PyArray_Check(obj))
      {
        PyArrayObject* const arr = reinterpret_cast<PyArrayObject*>(obj);
        int const ndim = PyArray_NDIM(arr);

        if (ndim != 3)
        {
          return NULL;
        }

        NPY_TYPES const expected_type = python_pixel_type<pixel_t>();
        int const type = PyArray_TYPE(arr);

        if (type != expected_type)
        {
          return NULL;
        }

        return obj;
      }

      return NULL;
    }

    static PyObject* convert(image_t const& img)
    {
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

      PyObject* const arr = PyArray_New(&PyArray_Type, nd, dims, arr_type, strides, reinterpret_cast<void*>(mem), 0, flags, obj);

      PyDimMem_FREE(dims);
      PyDimMem_FREE(strides);

      return arr;
    }

    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
      void* storage = reinterpret_cast<boost::python::converter::rvalue_from_python_storage<image_t>*>(data)->storage.bytes;

      PyArrayObject* const arr = reinterpret_cast<PyArrayObject*>(obj);
      npy_intp const* const dims = PyArray_DIMS(arr);
      npy_intp const* const strides = PyArray_STRIDES(arr);
      void* const mem = PyArray_DATA(arr);

      vil_memory_chunk_sptr chunk = new numpy_memory_chunk(arr);

      new (storage) image_t(chunk, reinterpret_cast<pixel_t*>(mem),
                            dims[0], dims[1], dims[2],
                            strides[0], strides[1], strides[2]);
      data->convertible = storage;
    }
};

template <typename T>
void register_vil_image_converter();

BOOST_PYTHON_MODULE(vil)
{
  import_array();

  // Expose vil_memory_chunk to Python. This is treated as opaque because Python
  // shouldn't be messing with such things, but it allows us to have numpy
  // arrays hold a reference to the memory chunk that is being used when
  // converting a vil_image_view into a NumPy array.
  class_<vil_memory_chunk, vil_memory_chunk_sptr, boost::noncopyable>("_VilMemoryChunk"
    , "<internal>"
    , no_init);

  register_vil_image_converter<vil_image_view<bool> >();
  register_vil_image_converter<vil_image_view<uint8_t> >();
  register_vil_image_converter<vil_image_view<float> >();
  register_vil_image_converter<vil_image_view<double> >();

  vistk::python::register_type<vil_image_view<bool> >(10);
  vistk::python::register_type<vil_image_view<uint8_t> >(10);
  vistk::python::register_type<vil_image_view<float> >(10);
  vistk::python::register_type<vil_image_view<double> >(10);
}

template <>
NPY_TYPES
python_pixel_type<bool>()
{
  return NPY_BOOL;
}

template <>
NPY_TYPES
python_pixel_type<uint8_t>()
{
  return NPY_UBYTE;
}

template <>
NPY_TYPES
python_pixel_type<float>()
{
  return NPY_FLOAT;
}

template <>
NPY_TYPES
python_pixel_type<double>()
{
  return NPY_DOUBLE;
}

template <typename T>
void
register_vil_image_converter()
{
  typedef vil_image_converter<T> converter_t;
  typedef typename converter_t::image_t image_t;

  boost::python::to_python_converter<image_t, converter_t>();
  converter_t();
}
