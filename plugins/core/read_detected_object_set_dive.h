/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_dive
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_DIVE_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_DIVE_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>

#include <memory>

namespace viame {

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
