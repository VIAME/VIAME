/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_VIL_TO_NUMPY_H
#define VISTK_PYTHON_NUMPY_VIL_TO_NUMPY_H

#include "numpy-config.h"

#include <vil/vil_image_view.h>
#include <vil/vil_image_view_base.h>

#include <Python.h>

/**
 * \file vil_to_numpy.h
 *
 * \brief Implementation of a vil-to-NumPy converter function.
 */

namespace vistk
{

namespace python
{

/**
 * \brief Convert a \c vil_image_view_base into a NumPy image.
 *
 * \returns A NumPy image, or \c None on failure.
 */
VISTK_PYTHON_NUMPY_EXPORT PyObject* vil_base_to_numpy(vil_image_view_base_sptr const& img);

/**
 * \brief Convert a \c vil_image_view instantation into a NumPy image.
 *
 * \returns A NumPy image, or \c None on failure.
 */
template <typename T>
VISTK_PYTHON_NUMPY_EXPORT PyObject* vil_to_numpy(vil_image_view<T> const& img);

}

}

#endif // VISTK_PYTHON_NUMPY_VIL_TO_NUMPY_H
