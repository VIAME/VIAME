/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_IMAGE_INSTANTIATE_H
#define VISTK_IMAGE_INSTANTIATE_H

#include <boost/cstdint.hpp>

/**
 * \file image/instantiate.h
 *
 * \brief Types for instantiation of image library templates.
 */

/**
 * \brief Integer pixel types for image algorithms.
 *
 * \param cls The class to instantiate.
 */
#define VISTK_IMAGE_INSTANTIATE_INT(cls) \
  template class cls<bool>;              \
  template class cls<uint8_t>

/**
 * \brief Floating point pixel types for image algorithms.
 *
 * \param cls The class to instantiate.
 */
#define VISTK_IMAGE_INSTANTIATE_FLOAT(cls) \
  template class cls<float>;               \
  template class cls<double>

/**
 * \brief All pixel types for image algorithms.
 *
 * \param cls The class to instantiate.
 */
#define VISTK_IMAGE_INSTANTIATE(cls) \
  VISTK_IMAGE_INSTANTIATE_INT(cls);  \
  VISTK_IMAGE_INSTANTIATE_FLOAT(cls)

#endif // VISTK_IMAGE_INSTANTIATE_H
