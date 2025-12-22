/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_AUTO_DETECT_TRANSFORM_H
#define VIAME_CORE_AUTO_DETECT_TRANSFORM_H

#include <plugins/core/viame_core_export.h>

#include <vital/algo/transform_2d_io.h>
#include <vital/types/transform_2d.h>


namespace viame
{

/// Automatically detect transform type and load it
class VIAME_CORE_EXPORT auto_detect_transform_io
  : public kwiver::vital::algo::transform_2d_io
{
public:

  static constexpr char const* name = "auto";

  static constexpr char const* description = "Automatically detect a transform "
    "type stored in either an ITK (.h5) or simple homography (.txt) format.";

  /// Constructor
  auto_detect_transform_io();

  /// Destructor
  virtual ~auto_detect_transform_io();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of transform_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns a transform instance referring to the loaded transform
   */
  virtual kwiver::vital::transform_2d_sptr load_(
    std::string const& filename ) const;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of transform_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param data the transform instance referring to the transform to write
   */
  virtual void save_( std::string const& filename,
                      kwiver::vital::transform_2d_sptr data ) const;

};

} // end namespace viame

#endif // VIAME_CORE_AUTO_DETECT_TRANSFORM_H
