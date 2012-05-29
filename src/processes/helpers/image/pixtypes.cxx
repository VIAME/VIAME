/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
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
::pixtype_bool()
{
  static pixtype_t const type = pixtype_t("bool");
  return type;
}

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

pixtype_t const&
pixtypes
::pixtype_double()
{
  static pixtype_t const type = pixtype_t("double");
  return type;
}

pixfmt_t const&
pixfmts
::pixfmt_mask()
{
  static pixfmt_t const fmt = pixfmt_t("mask");
  return fmt;
}

pixfmt_t const&
pixfmts
::pixfmt_rgb()
{
  static pixfmt_t const fmt = pixfmt_t("rgb");
  return fmt;
}

pixfmt_t const&
pixfmts
::pixfmt_bgr()
{
  static pixfmt_t const fmt = pixfmt_t("bgr");
  return fmt;
}

pixfmt_t const&
pixfmts
::pixfmt_rgba()
{
  static pixfmt_t const fmt = pixfmt_t("rgba");
  return fmt;
}

pixfmt_t const&
pixfmts
::pixfmt_bgra()
{
  static pixfmt_t const fmt = pixfmt_t("bgra");
  return fmt;
}

pixfmt_t const&
pixfmts
::pixfmt_yuv()
{
  static pixfmt_t const fmt = pixfmt_t("yuv");
  return fmt;
}

pixfmt_t const&
pixfmts
::pixfmt_gray()
{
  static pixfmt_t const fmt = pixfmt_t("gray");
  return fmt;
}

}
