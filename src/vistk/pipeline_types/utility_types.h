/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_TYPES_UTILITY_TYPES_H
#define VISTK_PIPELINE_TYPES_UTILITY_TYPES_H

#include "pipeline_types-config.h"

#include <vistk/pipeline/process.h>

/**
 * \file utility_types.h
 *
 * \brief Utility port types within the pipeline.
 */

namespace vistk
{

/**
 * \class utility_types utility_types.h <vistk/pipeline_types/utility_types.h>
 *
 * \brief Utility port types.
 */
class VISTK_PIPELINE_TYPES_EXPORT utility_types
{
  public:
    /// The type for timestamp data on a port.
    static process::port_type_t const t_timestamp;

    /// The type for homography transform data on a port.
    static process::port_type_t const t_transform;
    static process::port_type_t const t_image_to_image_homography;
    /// The type for a homography from an image to an arbitrary plane.
    static process::port_type_t const t_image_to_plane_homography;
    /// The type for a homography from an arbitrary plane to an image.
    static process::port_type_t const t_plane_to_image_homography;
    /// The type for a homography from an image to UTM coordinates.
    static process::port_type_t const t_image_to_utm_homography;
    /// The type for a homography from UTM coordinates to an image.
    static process::port_type_t const t_utm_to_image_homography;
    /// The type for a homography from an arbitrary plane to UTM coordinates.
    static process::port_type_t const t_plane_to_utm_homography;
    /// The type for a homography from UTM coordinates to an arbitrary plane.
    static process::port_type_t const t_utm_to_plane_homography;
};

}

#endif // VISTK_PIPELINE_TYPES_UTILITY_TYPES_H
