/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_WRAP_VIL_SMART_PTR_H
#define VISTK_PYTHON_NUMPY_WRAP_VIL_SMART_PTR_H

#include <boost/python/pointee.hpp>
#include <boost/get_pointer.hpp>

#include <vil/vil_smart_ptr.h>

namespace boost
{

namespace python
{

template <typename T>
inline
T*
get_pointer(vil_smart_ptr<T> const& p)
{
  return p.ptr();
}

template <typename T>
struct pointee<vil_smart_ptr<T> >
{
  typedef T type;
};

// Don't hide other get_pointer instances.
using boost::python::get_pointer;
using boost::get_pointer;

}

}

#endif // VISTK_PYTHON_NUMPY_WRAP_VIL_SMART_PTR_H
