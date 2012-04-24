/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_IMAGE_MACROS_H
#define VISTK_PROCESSES_HELPER_IMAGE_MACROS_H

#include <boost/cstdint.hpp>

/**
 * \file macros.h
 *
 * \brief Macros to help manage templates.
 */

#define TYPE_CHECK(type, name, function)     \
  if (pixtype == pixtypes::pixtype_##name()) \
  {                                          \
    return &function<type>;                  \
  }

#define SPECIFY_INT_FUNCTION(function)     \
  TYPE_CHECK(bool, bool, function)         \
  else TYPE_CHECK(uint8_t, byte, function)

#define SPECIFY_FUNCTION(function)         \
  SPECIFY_INT_FUNCTION(function)           \
  else TYPE_CHECK(float, float, function)  \
  else TYPE_CHECK(double, double, function)

#endif // VISTK_PROCESSES_HELPER_IMAGE_MACROS_H
