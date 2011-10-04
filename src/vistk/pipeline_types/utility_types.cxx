/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "utility_types.h"

/**
 * \file utility_types.cxx
 *
 * \brief Utility port types within the pipeline.
 */

namespace vistk
{

process::port_type_t const utility_types::t_timestamp = process::port_type_t("_timestamp");

static process::port_type_t const homog_prefix = process::port_type_t("_homog");
static process::port_type_t const image_plane = process::port_type_t("_image");
static process::port_type_t const plane_plane = process::port_type_t("_plane");
static process::port_type_t const utm_plane = process::port_type_t("_utm");

process::port_type_t const utility_types::t_transform = process::port_type_t("_transform");
process::port_type_t const utility_types::t_image_to_image_homography = homog_prefix + image_plane + image_plane;
process::port_type_t const utility_types::t_image_to_plane_homography = homog_prefix + image_plane + plane_plane;
process::port_type_t const utility_types::t_plane_to_image_homography = homog_prefix + plane_plane + image_plane;
process::port_type_t const utility_types::t_image_to_utm_homography = homog_prefix + image_plane + utm_plane;
process::port_type_t const utility_types::t_utm_to_image_homography = homog_prefix + utm_plane + image_plane;
process::port_type_t const utility_types::t_plane_to_utm_homography = homog_prefix + plane_plane + utm_plane;
process::port_type_t const utility_types::t_utm_to_plane_homography = homog_prefix + utm_plane + plane_plane;

}
