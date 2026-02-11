/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for write_detected_object_set_viame_csv
 */

#ifndef VIAME_CORE_WRITE_DETECTED_OBJECT_SET_VIAME_CSV_H
#define VIAME_CORE_WRITE_DETECTED_OBJECT_SET_VIAME_CSV_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_output.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame
{

class VIAME_CORE_EXPORT write_detected_object_set_viame_csv
  : public kwiver::vital::algo::detected_object_set_output
{
public:
  // NOTE: Keep description in sync with read_detected_object_set_viame_csv
  PLUGGABLE_IMPL(
    write_detected_object_set_viame_csv,
    "Detected object set writer using viame_csv format.\n\n"
    "  - Column(s) 1: Detection or Track ID\n"
    "  - Column(s) 2: Video or Image Identifier\n"
    "  - Column(s) 3: Unique Frame Identifier\n"
    "  - Column(s) 4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    "  - Column(s) 8: Detection Confidence\n"
    "  - Column(s) 9: Target Length (0 or less if uncomputed)\n"
    "  - Column(s) 10-11+: Repeated Species, Confidence Pairs\n",
    PARAM_DEFAULT(
      write_frame_number, bool,
      "Write a frame number for the unique frame ID field (as opposed to a string "
      "identifier) for column 3 in the output csv.",
      true ),
    PARAM_DEFAULT(
      stream_identifier, std::string,
      "Optional fixed video name over-ride to write to output column 2 in the csv.",
      "" ),
    PARAM_DEFAULT(
      model_identifier, std::string,
      "Model identifier string to write to the header or the csv.",
      "" ),
    PARAM_DEFAULT(
      version_identifier, std::string,
      "Version identifier string to write to the header or the csv.",
      "" ),
    PARAM_DEFAULT(
      frame_rate, std::string,
      "Frame rate string to write to the header or the csv.",
      "" ),
    PARAM_DEFAULT(
      mask_to_poly_tol, double,
      "Write segmentation masks when available as polygons with the specified "
      "relative tolerance for the conversion.  Set to a negative value to disable.",
      -1 ),
    PARAM_DEFAULT(
      mask_to_poly_points, int,
      "Write segmentation masks when available as polygons with the specified "
      "maximum number of points.  Set to a negative value to disable.",
      20 ),
    PARAM_DEFAULT(
      top_n_classes, unsigned,
      "Only print out this maximum number of classes (highest score first)",
      0 )
  )

  virtual ~write_detected_object_set_viame_csv() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void write_set( const kwiver::vital::detected_object_set_sptr set,
                          std::string const& image_name );

private:
  void initialize() override;

  // Runtime state (not config)
  bool m_first;
  int m_frame_number;
};

} // end namespace

#endif // VIAME_CORE_WRITE_DETECTED_OBJECT_SET_VIAME_CSV_H
