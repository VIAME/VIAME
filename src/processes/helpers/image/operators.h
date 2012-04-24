/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_IMAGE_OPERATORS_H
#define VISTK_PROCESSES_HELPER_IMAGE_OPERATORS_H

#include "format.h"

#include <vistk/pipeline/types.h>

#include <boost/function.hpp>

/**
 * \file operator.h
 *
 * \brief Types and functions for operations on images.
 */

namespace vistk
{

/// The type of a binary operator function on images.
typedef boost::function<datum_t (datum_t const&, datum_t const&)> binop_func_t;

/**
 * \brief A binary \c and operation on a pair of images.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to binary \c and \p pixtype images.
 */
binop_func_t and_for_pixtype(pixtype_t const& pixtype);

/**
 * \brief A binary \c or operation on a pair of images.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to binary \c or \p pixtype images.
 */
binop_func_t or_for_pixtype(pixtype_t const& pixtype);

/**
 * \brief A binary \c xor operation on a pair of images.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to binary \c xor \p pixtype images.
 */
binop_func_t xor_for_pixtype(pixtype_t const& pixtype);

}

#endif // VISTK_PROCESSES_HELPER_IMAGE_OPERATORS_H
