/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_NUMPY_TO_VIL_H
#define VISTK_PYTHON_NUMPY_NUMPY_TO_VIL_H

#include <vil/vil_image_view_base.h>

#include <Python.h>

namespace vistk
{

namespace python
{

vil_image_view_base_sptr numpy_to_vil(PyObject* obj);

}

}

#endif // VISTK_PYTHON_NUMPY_NUMPY_TO_VIL_H
