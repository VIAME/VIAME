/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_IMPORT_H
#define VISTK_PYTHON_NUMPY_IMPORT_H

#include "numpy-config.h"

/**
 * \file import.h
 *
 * \brief Declarations of functions for importing the NumPy module.
 */

namespace vistk
{

namespace python
{

/**
 * \brief Import the NumPy module.
 */
void VISTK_PYTHON_NUMPY_EXPORT import_numpy();

}

}

#endif // VISTK_PYTHON_NUMPY_IMPORT_H
