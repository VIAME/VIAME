/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_IMAGE_PIXTYPES_H
#define VISTK_PROCESSES_HELPER_IMAGE_PIXTYPES_H

#include <string>

/**
 * \file pixtypes.h
 *
 * \brief Types and functions to manage pixtypes in the pipeline.
 */

namespace vistk
{

/// The type for the type for pixels in an image.
typedef std::string pixtype_t;

/**
 * \class pixtypes "image_helper.h"
 *
 * \brief Names for common pixtypes.
 */
class pixtypes
{
  public:
    /**
     * \brief The pixtype for byte images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with bytes for pixels.
     */
    static pixtype_t const& pixtype_byte();
    /**
     * \brief The pixtype for floating images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with floats for pixels.
     */
    static pixtype_t const& pixtype_float();
};

}

#endif // VISTK_PROCESSES_HELPER_IMAGE_PIXTYPES_H
