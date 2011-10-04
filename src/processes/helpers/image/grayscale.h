/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_IMAGE_GRAYSCALE_H
#define VISTK_PROCESSES_HELPER_IMAGE_GRAYSCALE_H

#include "format.h"

#include <vistk/pipeline/types.h>

#include <boost/function.hpp>

/**
 * \file grayscale.h
 *
 * \brief Types and functions to help convert images to grayscale in the pipeline.
 */

namespace vistk
{

/// The type of a function which turns an image into grayscale.
typedef boost::function<datum_t (datum_t const&)> gray_func_t;

/**
 * \brief A grayscale conversion for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to converting \p pixtype images to grayscale.
 */
gray_func_t gray_for_pixtype(pixtype_t const& pixtype);

}

#endif // VISTK_PROCESSES_HELPER_IMAGE_GRAYSCALE_H
