/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_NUMPY_SUPPORT_H
#define VISTK_PYTHON_NUMPY_NUMPY_SUPPORT_H

#include <numpy/numpyconfig.h>

/**
 * \file numpy_support.h
 *
 * \brief Support macros for NumPy.
 */

/**
 * \def NPY
 *
 * \brief Compatability macro for NumPy.
 *
 * \param x The NumPy symbol.
 */

#if NPY_API_VERSION >= 0x00000007
// All NPY_* defines were renamed to NPY_ARRAY_* in 1.7 and the old ones
// deprecated.
#define NPY(x) NPY_ARRAY_##x
#else
#define NPY(x) NPY_##x
#endif

#endif // VISTK_PYTHON_NUMPY_SUPPORT_H
