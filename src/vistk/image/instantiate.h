/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <boost/cstdint.hpp>

#define VISTK_IMAGE_INSTANTIATE(cls) \
  template class cls<bool>;          \
  template class cls<uint8_t>;       \
  template class cls<float>;         \
  template class cls<double>
