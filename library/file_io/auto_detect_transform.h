/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_FILE_IO_AUTO_DETECT_TRANSFORM_H
#define VIAME_FILE_IO_AUTO_DETECT_TRANSFORM_H

#include "viame_file_io_export.h"

#include <viame/algorithm_framework/algo/transform_2d_io.h>
#include <viame/core_types/transform_2d.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>


namespace viame
{

/// Automatically detect transform type and load it
class VIAME_FILE_IO_EXPORT auto_detect_transform_io
  : public viame::algo::transform_2d_io
{
public:
  PLUGGABLE_IMPL_NAMED(
    auto_detect_transform_io, "auto",
    "Automatically detect a transform type stored in either a DIVE camera "
    "registration (.json) or simple homography (.txt) format. ITK (.h5) "
    "transforms are rejected with a pointer to "
    "tools/convert_itk.py." )
  virtual ~auto_detect_transform_io() = default;

  virtual bool check_configuration( viame::config_block_sptr config ) const override;
private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of transform_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns a transform instance referring to the loaded transform
   */
  virtual viame::transform_2d_sptr load_(
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
                      viame::transform_2d_sptr data ) const;

};

} // end namespace viame

#endif // VIAME_FILE_IO_AUTO_DETECT_TRANSFORM_H
