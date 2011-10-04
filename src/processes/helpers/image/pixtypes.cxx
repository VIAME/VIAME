/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pixtypes.h"

/**
 * \file pixtypes.cxx
 *
 * \brief Implementations of functions to help manage pixtypes in the pipeline.
 */

namespace vistk
{

pixtype_t const&
pixtypes
::pixtype_byte()
{
  static pixtype_t const type = pixtype_t("byte");
  return type;
}

pixtype_t const&
pixtypes
::pixtype_float()
{
  static pixtype_t const type = pixtype_t("float");
  return type;
}

}
