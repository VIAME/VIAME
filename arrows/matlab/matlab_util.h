// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for MatLab util functions
 */

#ifndef ARROWS_MATLAB_UTIL_H
#define ARROWS_MATLAB_UTIL_H

#include <vital/types/image_container.h>

#include <arrows/matlab/mxarray.h>
#include <arrows/matlab/kwiver_algo_matlab_export.h>

namespace kwiver {
namespace arrows {
namespace matlab {

/**
 * @brief Convert image_container to matlab image
 *
 * This function converts the supplied image from the container to a
 * matlab compatible image. The image pixels are copied to the matlab
 * image memory.
 *
 * @param in_image Input image container.
 *
 * @return Managed array containing the image.
 */
KWIVER_ALGO_MATLAB_EXPORT
MxArraySptr convert_mx_image( const kwiver::vital::image_container_sptr image );

KWIVER_ALGO_MATLAB_EXPORT
kwiver::vital::image_container_sptr convert_mx_image( const MxArraySptr image );

/** \defgroup create_matlab_array Create Matlab Array
 * Factory functions to create managed Matlab arrays.
 * @{
 */

/**
 * @brief Create empty Matlab managed array.
 *
 * This function is a factory for managed Matlab arrays.
 *
 * @param r - number of rows in the array
 * @param c - number of columns in the array
 *
 * @return Managed pointer to the newly allocated array.
 */
KWIVER_ALGO_MATLAB_EXPORT
MxArraySptr create_mxByteArray( size_t r, size_t c );

/**
 * @brief Create empty Matlab managed array.
 *
 * This function is a factory for managed Matlab arrays.
 *
 * @param r - number of rows in the array
 * @param c - number of columns in the array
 *
 * @return Managed pointer to the newly allocated array.
 */
KWIVER_ALGO_MATLAB_EXPORT
MxArraySptr create_mxIntArray( size_t r, size_t c );

/**
 * @brief Create empty Matlab managed array.
 *
 * This function is a factory for managed Matlab arrays.
 *
 * @param r - number of rows in the array
 * @param c - number of columns in the array
 *
 * @return Managed pointer to the newly allocated array.
 */
KWIVER_ALGO_MATLAB_EXPORT
MxArraySptr create_mxDoubleArray( size_t r, size_t c );
//@}

} } } // end namespace

#endif // ARROWS_MATLAB_UTIL_H
