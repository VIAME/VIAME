/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_IMAGE_WRITE_H
#define VISTK_PROCESSES_HELPER_IMAGE_WRITE_H

#include "format.h"

#include <vistk/pipeline_utils/types.h>

#include <vistk/pipeline/types.h>

#include <boost/function.hpp>

/**
 * \file write.h
 *
 * \brief Types and functions to help write images within the pipeline.
 */

namespace vistk
{

/// The type of a function which writes in an image to a file.
typedef boost::function<void (path_t const&, datum_t const&)> write_func_t;

/**
 * \brief A writing function for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to writing \p pixtype images to a file.
 */
write_func_t write_for_pixtype(pixtype_t const& pixtype);

}

#endif // VISTK_PROCESSES_HELPER_IMAGE_WRITE_H
