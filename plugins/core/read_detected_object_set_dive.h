/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_dive and shared DIVE parsing
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_DIVE_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_DIVE_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/bounding_box.h>
#include <vital/logger/logger.h>

#include <memory>
#include <vector>
#include <map>
#include <string>

namespace viame {

// -----------------------------------------------------------------------------------
// DIVE JSON data structures (shared between detection and track readers)
// -----------------------------------------------------------------------------------

/// \brief A single feature (detection) within a DIVE track
struct VIAME_CORE_EXPORT dive_feature
{
  int frame = 0;
  std::vector< double > bounds;  // [x1, y1, x2, y2]
  bool keyframe = false;
  bool interpolate = false;
  std::vector< double > head;
  std::vector< double > tail;
  double fishLength = 0.0;
  std::map< std::string, std::string > attributes;
};

/// \brief A track in DIVE format containing multiple features
struct VIAME_CORE_EXPORT dive_track
{
  int id = 0;
  int begin = 0;
  int end = 0;
  std::vector< std::pair< std::string, double > > confidencePairs;
  std::vector< dive_feature > features;
  std::map< std::string, std::string > attributes;
};

/// \brief Top-level DIVE annotation file structure
struct VIAME_CORE_EXPORT dive_annotation_file
{
  std::map< std::string, dive_track > tracks;
  int version = 1;
};

// -----------------------------------------------------------------------------------
// DIVE JSON parsing utilities (shared between detection and track readers)
// -----------------------------------------------------------------------------------

/// \brief Parse a DIVE JSON file
///
/// \param filename Path to the DIVE JSON file
/// \param logger Logger for error/warning messages
/// \param[out] dive_data Parsed annotation data
/// \return true if parsing succeeded, false otherwise
VIAME_CORE_EXPORT
bool parse_dive_json_file( std::string const& filename,
                           kwiver::vital::logger_handle_t logger,
                           dive_annotation_file& dive_data );

/// \brief Manual fallback parser for DIVE JSON format
///
/// \param content JSON content as string
/// \param logger Logger for messages
/// \param[out] dive_data Parsed annotation data
/// \return true if parsing succeeded, false otherwise
VIAME_CORE_EXPORT
bool parse_dive_json_manual( std::string const& content,
                             kwiver::vital::logger_handle_t logger,
                             dive_annotation_file& dive_data );

/// \brief Create a detected_object from a DIVE feature and track confidence pairs
///
/// \param feature The DIVE feature containing bounds
/// \param confidence_pairs The track's confidence pairs for class labels
/// \return A detected_object_sptr, or nullptr if feature has no valid bounds
VIAME_CORE_EXPORT
kwiver::vital::detected_object_sptr
create_detected_object_from_dive(
  dive_feature const& feature,
  std::vector< std::pair< std::string, double > > const& confidence_pairs );

// -----------------------------------------------------------------------------------
// Detection reader class
// -----------------------------------------------------------------------------------

/// \brief Read detected object sets from DIVE JSON format.
///
/// DIVE JSON format contains:
///   - tracks: object with track ID keys containing track data
///   - Each track has: id, begin, end, confidencePairs, features
///   - Each feature has: frame, bounds [x1,y1,x2,y2], attributes
///   - Optional geometry field with GeoJSON polygons
///
/// See: https://kitware.github.io/dive/DataFormats/
///
class VIAME_CORE_EXPORT read_detected_object_set_dive
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  static constexpr char const* name = "dive";

  static constexpr char const* description =
    "Detected object set reader using DIVE JSON format.\n\n"
    "DIVE JSON format contains:\n"
    "  - tracks: object with track data indexed by ID\n"
    "  - Each track: id, begin, end, confidencePairs, features\n"
    "  - Each feature: frame, bounds [x1,y1,x2,y2], geometry\n"
    "  - confidencePairs: [[label, score], ...]\n\n"
    "See: https://kitware.github.io/dive/DataFormats/";

  read_detected_object_set_dive();
  virtual ~read_detected_object_set_dive();

  virtual kwiver::vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_DIVE_H
