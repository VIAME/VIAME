/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "operators.h"

#include "macros.h"

#include <vistk/pipeline/datum.h>

#include <vil/vil_image_view.h>
#include <vil/vil_transform.h>

/**
 * \file operator.cxx
 *
 * \brief Implementations of functions for operations on images.
 */

namespace vistk
{

template <typename PixType>
static datum_t and_(datum_t const& lhs, datum_t const& rhs);
template <typename PixType>
static datum_t or_(datum_t const& lhs, datum_t const& rhs);
template <typename PixType>
static datum_t xor_(datum_t const& lhs, datum_t const& rhs);

binop_func_t
and_for_pixtype(pixtype_t const& pixtype)
{
  SPECIFY_INT_FUNCTION(and_)

  return binop_func_t();
}

binop_func_t
or_for_pixtype(pixtype_t const& pixtype)
{
  SPECIFY_INT_FUNCTION(or_)

  return binop_func_t();
}

binop_func_t
xor_for_pixtype(pixtype_t const& pixtype)
{
  SPECIFY_INT_FUNCTION(xor_)

  return binop_func_t();
}

template <typename PixType>
struct pixel_function
{
  typedef boost::function<PixType (PixType const&)> return_unary_function_t;
  typedef boost::function<void (PixType&)> mutate_unary_function_t;

  typedef boost::function<PixType (PixType const&, PixType const&)> return_binary_function_t;
  typedef boost::function<void (PixType const&, PixType&)> mutate_binary_function_t;
};

template <typename PixType>
static datum_t pixel_binary_operation(datum_t const& lhs, datum_t const& rhs,
                                      typename pixel_function<PixType>::return_binary_function_t const& function);

template <typename PixType>
static PixType pixel_and(PixType const& lhs, PixType const& rhs);
template <typename PixType>
static PixType pixel_or(PixType const& lhs, PixType const& rhs);
template <typename PixType>
static PixType pixel_xor(PixType const& lhs, PixType const& rhs);

template <typename PixType>
datum_t
and_(datum_t const& lhs, datum_t const& rhs)
{
  return pixel_binary_operation<PixType>(lhs, rhs, &pixel_and<PixType>);
}

template <typename PixType>
datum_t
or_(datum_t const& lhs, datum_t const& rhs)
{
  return pixel_binary_operation<PixType>(lhs, rhs, &pixel_or<PixType>);
}

template <typename PixType>
datum_t
xor_(datum_t const& lhs, datum_t const& rhs)
{
  return pixel_binary_operation<PixType>(lhs, rhs, &pixel_xor<PixType>);
}

template <typename PixType>
PixType
pixel_and(PixType const& lhs, PixType const& rhs)
{
  return (lhs & rhs);
}

template <typename PixType>
PixType
pixel_or(PixType const& lhs, PixType const& rhs)
{
  return (lhs | rhs);
}

template <typename PixType>
PixType
pixel_xor(PixType const& lhs, PixType const& rhs)
{
  return (lhs ^ rhs);
}

template <typename PixType>
datum_t
pixel_binary_operation(datum_t const& lhs, datum_t const& rhs,
                       typename pixel_function<PixType>::return_binary_function_t const& function)
{
  if (!lhs || (lhs->type() == datum::empty))
  {
    return rhs;
  }

  typedef vil_image_view<PixType> image_t;

  image_t const l = lhs->get_datum<image_t>();
  image_t const r = rhs->get_datum<image_t>();

  /// \todo Sanity check the parameters.

  image_t img;

  vil_transform(l, r, img, function);

  if (!r)
  {
    static datum::error_t const err_string = datum::error_t("Unable to 'or' the images.");

    return datum::error_datum(err_string);
  }

  return datum::new_datum(img);
}

}
