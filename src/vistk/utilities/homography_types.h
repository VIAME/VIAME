/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_HOMOGRAPHY_TYPES_H
#define VISTK_UTILITIES_HOMOGRAPHY_TYPES_H

#include "utilities-config.h"

#include "homography.h"
#include "plane_ref.h"
#include "timestamp.h"
#include "utm.h"

/**
 * \file homography_types.h
 *
 * \brief Typedefs for common homography sources and destinations.
 */

namespace vistk
{

/// A homography from one image to another.
typedef homography<timestamp, timestamp>    image_to_image_homography;
/// A homography from image to a reference plane.
typedef homography<timestamp, plane_ref_t>  image_to_plane_homography;
/// A homography from reference plane to an image.
typedef homography<plane_ref_t, timestamp>  plane_to_image_homography;
/// A homography from image to the UTM coordinate space.
typedef homography<timestamp, utm_zone_t>   image_to_utm_homography;
/// A homography from the UTM coordinate space to an image.
typedef homography<utm_zone_t, timestamp>   utm_to_image_homography;
/// A homography from reference plane to the UTM coodinate space.
typedef homography<plane_ref_t, utm_zone_t> plane_to_utm_homography;
/// A homography from the UTM coodinate space to a reference plane.
typedef homography<utm_zone_t, plane_ref_t> utm_to_plane_homography;

}

#endif // VISTK_UTILITIES_HOMOGRAPHY_TYPES_H
