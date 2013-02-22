/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "python_wrap_vil_smart_ptr.h"

#include <boost/python/class.hpp>
#include <boost/thread/once.hpp>

#include <vil/vil_image_view.h>
#include <vil/vil_image_view_base.h>
#include <vil/vil_memory_chunk.h>

#include <typeinfo>

/**
 * \file numpy/registration.cxx
 *
 * \brief Implementation of functions to register vil types to \c boost::python.
 */

using namespace boost::python;

namespace vistk
{

namespace python
{

static void register_memory_chunk_to_python();

void
register_memory_chunk()
{
  static boost::once_flag once;

  boost::call_once(once, register_memory_chunk_to_python);
}

void
register_memory_chunk_to_python()
{
  // Expose vil_memory_chunk to Python. This is treated as opaque because Python
  // shouldn't be messing with such things, but it allows us to have numpy
  // arrays hold a reference to the memory chunk that is being used when
  // converting a vil_image_view into a NumPy array.
  class_<vil_memory_chunk, vil_memory_chunk_sptr>("_VilMemoryChunk"
    , "<internal>"
    , no_init);
}

static void register_image_base_to_python();

void
register_image_base()
{
  static boost::once_flag once;

  boost::call_once(once, register_image_base_to_python);
}

void
register_image_base_to_python()
{
  class_<vil_image_view_base_sptr>("_VilImage"
    , "<internal>"
    , no_init);
}

template <typename T>
static void register_image_to_python();

template <typename T>
void
register_image_type()
{
  static boost::once_flag once;

  boost::call_once(once, register_image_to_python<T>);
}

template <typename T>
void
register_image_to_python()
{
  std::string base = "_VilImage_";

  base += typeid(T).name();

  class_<vil_image_view<T> >(base.c_str()
    , "<internal>"
    , no_init);
}

}

}

#define INSTANTIATE_REGISTER_IMAGE_TYPE(type) \
  template VISTK_PYTHON_NUMPY_EXPORT void vistk::python::register_image_type<type>()

INSTANTIATE_REGISTER_IMAGE_TYPE(bool);
INSTANTIATE_REGISTER_IMAGE_TYPE(signed char);
INSTANTIATE_REGISTER_IMAGE_TYPE(unsigned char);
INSTANTIATE_REGISTER_IMAGE_TYPE(short);
INSTANTIATE_REGISTER_IMAGE_TYPE(unsigned short);
INSTANTIATE_REGISTER_IMAGE_TYPE(int);
INSTANTIATE_REGISTER_IMAGE_TYPE(unsigned int);
INSTANTIATE_REGISTER_IMAGE_TYPE(long);
INSTANTIATE_REGISTER_IMAGE_TYPE(unsigned long);
#if VXL_HAS_INT_64
INSTANTIATE_REGISTER_IMAGE_TYPE(long long);
INSTANTIATE_REGISTER_IMAGE_TYPE(unsigned long long);
#endif
INSTANTIATE_REGISTER_IMAGE_TYPE(float);
INSTANTIATE_REGISTER_IMAGE_TYPE(double);

#undef INSTANTIATE_REGISTER_IMAGE_TYPE
