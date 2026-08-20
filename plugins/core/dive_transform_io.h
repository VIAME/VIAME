/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_DIVE_TRANSFORM_IO_H
#define VIAME_CORE_DIVE_TRANSFORM_IO_H

#include "viame_core_export.h"

#include <vital/algo/transform_2d_io.h>
#include <vital/types/transform_2d.h>

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
class VIAME_CORE_EXPORT dive_transform_io
  : public kwiver::vital::algo::transform_2d_io
{
public:

  static constexpr char const* name = "dive";

  static constexpr char const* description = "Reads a homography from a DIVE "
    "camera registration (.json) file. When the file contains multiple camera "
    "pairs, or the desired direction is the reverse of the stored pair, set "
    "from_camera and to_camera to select the transform mapping points from "
    "from_camera image coordinates into to_camera image coordinates.";

  /// Constructor
  dive_transform_io();

  /// Destructor
  virtual ~dive_transform_io();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

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

  /// Camera whose image coordinates the loaded transform maps from
  std::string m_from_camera;
  /// Camera whose image coordinates the loaded transform maps into
  std::string m_to_camera;
};

} // end namespace viame

#endif // VIAME_CORE_DIVE_TRANSFORM_IO_H
