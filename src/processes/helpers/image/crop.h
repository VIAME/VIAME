/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_IMAGE_CROP_H
#define VISTK_PROCESSES_HELPER_IMAGE_CROP_H

#include "format.h"

#include <vistk/pipeline/types.h>

#include <boost/function.hpp>

/**
 * \file crop.h
 *
 * \brief Types and functions to help cropping images in the pipeline.
 */

namespace vistk
{

/// The type of a function which crops an image.
typedef boost::function<datum_t (datum_t const&, size_t, size_t, size_t, size_t)> crop_func_t;

/**
 * \brief A cropping function for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to cropping \p pixtype images.
 */
crop_func_t crop_for_pixtype(pixtype_t const& pixtype);

}

#endif // VISTK_PROCESSES_HELPER_IMAGE_CROP_H
