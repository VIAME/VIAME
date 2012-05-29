/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_CONFIG_H_
#define VISTK_PYTHON_NUMPY_CONFIG_H_

#include <vistk/config.h>

/**
 * \file numpy-config.h
 *
 * \brief Defines for symbol visibility in numpy.
 */

#ifndef VISTK_PYTHON_NUMPY_EXPORT
#ifdef MAKE_VISTK_PYTHON_NUMPY_LIB
/// Export the symbol if building the library.
#define VISTK_PYTHON_NUMPY_EXPORT VISTK_EXPORT
#else
/// Import the symbol if including the library.
#define VISTK_PYTHON_NUMPY_EXPORT VISTK_IMPORT
#endif // MAKE_VISTK_PYTHON_NUMPY_LIB
/// Hide the symbol from the library interface.
#define VISTK_PYTHON_NUMPY_NO_EXPORT VISTK_NO_EXPORT
#endif // VISTK_PYTHON_NUMPY_EXPORT

#ifndef VISTK_PYTHON_NUMPY_EXPORT_DEPRECATED
/// Mark as deprecated.
#define VISTK_PYTHON_NUMPY_EXPORT_DEPRECATED VISTK_DEPRECATED VISTK_PYTHON_NUMPY_EXPORT
#endif // VISTK_PYTHON_NUMPY_EXPORT_DEPRECATED

#endif // VISTK_PYTHON_NUMPY_CONFIG_H_
