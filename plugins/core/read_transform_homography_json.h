/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_READ_TRANSFORM_HOMOGRAPHY_JSON_H
#define VIAME_CORE_READ_TRANSFORM_HOMOGRAPHY_JSON_H

#include "viame_core_export.h"

#include <vital/algo/transform_2d_io.h>
#include <vital/types/transform_2d.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <string>


namespace viame
{

/// Read and write homographies in the DIVE camera registration JSON format.
///
/// DIVE (the annotator) saves camera-to-camera registrations as
/// "dive-camera-registration" JSON files containing one or more camera
/// pairs, each with picked point correspondences and fitted 3x3 homographies
/// in both directions ("leftToRight" / "rightToLeft", row-major, either of
/// which may be null when unfitted). This reader selects one pair and
/// direction and returns it as a homography transform.
class VIAME_CORE_EXPORT read_transform_homography_json
  : public kwiver::vital::algo::transform_2d_io
{
public:
  PLUGGABLE_IMPL_NAMED(
    read_transform_homography_json, "homography_json",
    "Reads a homography from a DIVE camera registration (.json) file. When "
    "the file contains multiple camera pairs, or the desired direction is the "
    "reverse of the stored pair, set from_camera and to_camera to select the "
    "transform mapping points from from_camera image coordinates into "
    "to_camera image coordinates.",
    PARAM_DEFAULT(
      from_camera, std::string,
      "Camera whose image coordinates the transform maps from. Leave empty "
      "with to_camera to use the single pair in the file, left to right.",
      "" ),
    PARAM_DEFAULT(
      to_camera, std::string,
      "Camera whose image coordinates the transform maps into. Leave empty "
      "with from_camera to use the single pair in the file, left to right.",
      "" ) )

  virtual ~read_transform_homography_json() = default;

  virtual bool check_configuration(
    kwiver::vital::config_block_sptr config ) const override;

private:
  /// Implementation specific load functionality.
  /**
   * \param filename the path to the file the load
   * \returns a transform instance referring to the loaded transform
   */
  virtual kwiver::vital::transform_2d_sptr load_(
    std::string const& filename ) const;

  /// Implementation specific save functionality.
  /**
   * \param filename the path to the file to save
   * \param data the transform instance referring to the transform to write
   */
  virtual void save_( std::string const& filename,
                      kwiver::vital::transform_2d_sptr data ) const;
};

} // end namespace viame

#endif // VIAME_CORE_READ_TRANSFORM_HOMOGRAPHY_JSON_H
