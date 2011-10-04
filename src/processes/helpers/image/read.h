/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_IMAGE_READ_H
#define VISTK_PROCESSES_HELPER_IMAGE_READ_H

#include "format.h"

#include <vistk/utilities/path.h>

#include <vistk/pipeline/types.h>

#include <boost/function.hpp>

/**
 * \file read.h
 *
 * \brief Types and functions to help read images within the pipeline.
 */

namespace vistk
{

/// The type of a function which reads an image from a file.
typedef boost::function<datum_t (path_t const&)> read_func_t;

/**
 * \brief A reading function for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to reading \p pixtype images from a file.
 */
read_func_t read_for_pixtype(pixtype_t const& pixtype);

}

#endif // VISTK_PROCESSES_HELPER_IMAGE_READ_H
