/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_NUMPY_TO_VIL_H
#define VISTK_PYTHON_NUMPY_NUMPY_TO_VIL_H

#include "numpy-config.h"

#include <vil/vil_image_view.h>
#include <vil/vil_image_view_base.h>

#include <Python.h>

/**
 * \file numpy_to_vil.h
 *
 * \brief Declaration of a NumPy-to-vil converter function.
 */

namespace vistk
{

namespace python
{

/**
 * \brief Convert a NumPy object into a base vil image.
 *
 * \returns A vil image, or \c NULL on failure.
 */
VISTK_PYTHON_NUMPY_EXPORT vil_image_view_base_sptr numpy_to_vil_base(PyObject* obj);

/**
 * \brief Convert a NumPy object into a typed vil image.
 *
 * \returns A vil image, or an empty view on failure.
 */
template <typename T>
VISTK_PYTHON_NUMPY_EXPORT vil_image_view<T> numpy_to_vil(PyObject* obj);

}

}

#endif // VISTK_PYTHON_NUMPY_NUMPY_TO_VIL_H
