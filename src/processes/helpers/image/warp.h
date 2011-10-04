/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_IMAGE_WARP_H
#define VISTK_PROCESSES_HELPER_IMAGE_WARP_H

#include "format.h"

#include <vistk/pipeline/types.h>

#include <boost/function.hpp>

/**
 * \file warp.h
 *
 * \brief Types and functions to help warp images within the pipeline.
 */

namespace vistk
{

/// The type of a function which warps an image.
typedef boost::function<datum_t (datum_t const&, datum_t const&)> warp_func_t;

/**
 * \brief An image warping function for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to warping \p pixtype images.
 */
warp_func_t warp_for_pixtype(pixtype_t const& pixtype);

}

#endif // VISTK_PROCESSES_HELPER_IMAGE_WARP_H
