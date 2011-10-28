/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_TYPES_IMAGE_TYPES_H
#define VISTK_PIPELINE_TYPES_IMAGE_TYPES_H

#include "pipeline_types-config.h"

#include <vistk/pipeline/process.h>

/**
 * \file image_types.h
 *
 * \brief Image port types within the pipeline.
 */

namespace vistk
{

/**
 * \class image_types image_types.h <vistk/pipeline_types/image_types.h>
 *
 * \brief Image port types.
 */
class VISTK_PIPELINE_TYPES_EXPORT image_types
{
  public:
    /// The type for 1-bit masks.
    static process::port_type_t const t_bitmask;
    /// The type for 8-bit masks.
    static process::port_type_t const t_bytemask;

    /// The type for 8-bit grayscale images.
    static process::port_type_t const t_byte_grayscale;
    /// The type for floating point grayscale images.
    static process::port_type_t const t_float_grayscale;

    /// The type for 8-bit RGB images.
    static process::port_type_t const t_byte_rgb;
    /// The type for floating point RGB images.
    static process::port_type_t const t_float_rgb;

    /// The type for 8-bit BGR images.
    static process::port_type_t const t_byte_bgr;
    /// The type for floating point BGR images.
    static process::port_type_t const t_float_bgr;

    /// The type for 8-bit RGBA images.
    static process::port_type_t const t_byte_rgba;
    /// The type for floating point RGBA images.
    static process::port_type_t const t_float_rgba;

    /// The type for 8-bit BGRA images.
    static process::port_type_t const t_byte_bgra;
    /// The type for floating point BGRA images.
    static process::port_type_t const t_float_bgra;

    /// The type for 8-bit YUV images.
    static process::port_type_t const t_byte_yuv;
    /// The type for floating point YUV images.
    static process::port_type_t const t_float_yuv;

    /// The type for IPL images.
    static process::port_type_t const t_ipl;
};

}

#endif // VISTK_PIPELINE_TYPES_IMAGE_TYPES_H
