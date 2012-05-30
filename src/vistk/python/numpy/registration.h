/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_NUMPY_REGISTRATION_H
#define VISTK_PYTHON_NUMPY_REGISTRATION_H

#include "numpy-config.h"

/**
 * \file numpy/registration.h
 *
 * \brief Declaration of functions to register vil types to \c boost::python.
 */

namespace vistk
{

namespace python
{

/**
 * \brief Register a \c vil_memory_chunk to Python.
 */
void VISTK_PYTHON_NUMPY_EXPORT register_memory_chunk();

/**
 * \brief Register a \c vil_image_view_base to Python.
 */
void VISTK_PYTHON_NUMPY_EXPORT register_image_base();

/**
 * \brief Register a \c vil_image_view instantiation to Python.
 */
template <typename T>
void VISTK_PYTHON_NUMPY_EXPORT register_image_type();

}

}

#endif // VISTK_PYTHON_NUMPY_REGISTRATION_H
