/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_VIL_TO_NUMPY_H
#define VISTK_PYTHON_NUMPY_VIL_TO_NUMPY_H

#include <vil/vil_image_view.h>
#include <vil/vil_image_view_base.h>

#include <Python.h>

namespace vistk
{

namespace python
{

PyObject* vil_to_numpy(vil_image_view_base_sptr const& img);

template <typename T>
PyObject* vil_to_numpy(vil_image_view<T> const& img);

}

}

#endif // VISTK_PYTHON_NUMPY_VIL_TO_NUMPY_H
