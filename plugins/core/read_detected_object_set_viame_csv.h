/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_viame_csv and shared CSV utilities
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_VIAME_CSV_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_VIAME_CSV_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/bounding_box.h>

#include <memory>
#include <vector>
#include <string>

namespace viame {

// =============================================================================
// VIAME CSV column definitions (shared between readers and writers)
// =============================================================================

/// Column indices for VIAME CSV format
enum viame_csv_column
{
  VIAME_CSV_COL_DET_ID = 0,   ///< Detection or Track ID
  VIAME_CSV_COL_SOURCE_ID,    ///< Video or Image Identifier
  VIAME_CSV_COL_FRAME_ID,     ///< Unique Frame Identifier
  VIAME_CSV_COL_MIN_X,        ///< Bounding box top-left X
  VIAME_CSV_COL_MIN_Y,        ///< Bounding box top-left Y
  VIAME_CSV_COL_MAX_X,        ///< Bounding box bottom-right X
  VIAME_CSV_COL_MAX_Y,        ///< Bounding box bottom-right Y
  VIAME_CSV_COL_CONFIDENCE,   ///< Detection confidence
  VIAME_CSV_COL_LENGTH,       ///< Target length (0 or less if uncomputed)
  VIAME_CSV_COL_TOT           ///< First optional column (species/confidence pairs start here)
};

// =============================================================================
// VIAME CSV parsing utilities (shared between detection and track readers)
// =============================================================================

/// Create a bounding box from VIAME CSV columns
///
/// \param cols Vector of CSV column values
/// \returns Bounding box created from columns 3-6
VIAME_CORE_EXPORT
kwiver::vital::bounding_box_d
create_viame_csv_bbox( std::vector< std::string > const& cols );

/// Parse species/confidence pairs from VIAME CSV columns
///
/// Parses columns starting at VIAME_CSV_COL_TOT, reading pairs of
/// (species_name, confidence) until an optional field marker '(' is found
/// or end of columns.
///
/// \param cols Vector of CSV column values
/// \param confidence_override If > 0, use this value for all confidences
/// \param[out] dot Detected object type to populate with scores
/// \returns Index of first optional field column, or cols.size() if none found
VIAME_CORE_EXPORT
size_t parse_viame_csv_species(
  std::vector< std::string > const& cols,
  double confidence_override,
  kwiver::vital::detected_object_type_sptr& dot );

/// Extract polygon vertices from VIAME CSV optional fields
///
/// Searches columns for "(poly)" or "(+poly)" markers and extracts
/// the polygon vertex coordinates.
///
/// \param cols Vector of CSV column values
/// \param start_col Column index to start searching from
/// \param[out] polygon Output vector of polygon vertices (x1,y1,x2,y2,...)
/// \returns true if a polygon was found and extracted
VIAME_CORE_EXPORT
bool extract_viame_csv_polygon(
  std::vector< std::string > const& cols,
  size_t start_col,
  std::vector< double >& polygon );

/// Create a detected object from VIAME CSV columns
///
/// This is a convenience function that combines bbox creation, species parsing,
/// and polygon extraction into a single call.
///
/// \param cols Vector of CSV column values
/// \param confidence_override If > 0, use this value for all confidences
/// \returns Created detected object, or nullptr on error
VIAME_CORE_EXPORT
kwiver::vital::detected_object_sptr
create_viame_csv_detection(
  std::vector< std::string > const& cols,
  double confidence_override = -1.0 );

// =============================================================================
// VIAME CSV reader class
// =============================================================================

class VIAME_CORE_EXPORT read_detected_object_set_viame_csv
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  static constexpr char const* name = "viame_csv";

  // NOTE: Keep description in sync with write_detected_object_set_viame_csv
  static constexpr char const* description =
    "Detected object set reader using viame_csv format.\n\n"
    "  - Column(s) 1: Detection or Track ID\n"
    "  - Column(s) 2: Video or Image Identifier\n"
    "  - Column(s) 3: Unique Frame Identifier\n"
    "  - Column(s) 4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    "  - Column(s) 8: Detection Confidence\n"
    "  - Column(s) 9: Target Length (0 or less if uncomputed)\n"
    "  - Column(s) 10-11+: Repeated Species, Confidence Pairs\n";

  read_detected_object_set_viame_csv();
  virtual ~read_detected_object_set_viame_csv();

  virtual void set_configuration(kwiver::vital::config_block_sptr config);
  virtual bool check_configuration(kwiver::vital::config_block_sptr config) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_VIAME_CSV_H
