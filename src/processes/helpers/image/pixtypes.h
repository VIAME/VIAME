/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
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

/// The type for the format of pixels in an image.
typedef std::string pixfmt_t;

/**
 * \class pixtypes "pixtypes.h"
 *
 * \brief Names for common pixtypes.
 */
class pixtypes
{
  public:
    /**
     * \brief The pixtype for boolean images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with bools for pixels.
     */
    static pixtype_t const& pixtype_bool();
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
    /**
     * \brief The pixtype for double precision images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with doubles for pixels.
     */
    static pixtype_t const& pixtype_double();
};

/**
 * \class pixfmts "pixtypes.h"
 *
 * \brief Names for common pixfmts.
 */
class pixfmts
{
  public:
    /**
     * \brief The pixfmt for mask images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with mask pixels.
     */
    static pixfmt_t const& pixfmt_mask();
    /**
     * \brief The pixfmt for RGB images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with RGB pixels.
     */
    static pixfmt_t const& pixfmt_rgb();
    /**
     * \brief The pixfmt for BGR images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with BGR pixels.
     */
    static pixfmt_t const& pixfmt_bgr();
    /**
     * \brief The pixfmt for RGBA images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with RGBA pixels.
     */
    static pixfmt_t const& pixfmt_rgba();
    /**
     * \brief The pixfmt for BGRA images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with BGRA pixels.
     */
    static pixfmt_t const& pixfmt_bgra();
    /**
     * \brief The pixfmt for YUV images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with YUV pixels.
     */
    static pixfmt_t const& pixfmt_yuv();
    /**
     * \brief The pixfmt for grayscale images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with grayscale pixels.
     */
    static pixfmt_t const& pixfmt_gray();
};

}

#endif // VISTK_PROCESSES_HELPER_IMAGE_PIXTYPES_H
