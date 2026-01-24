/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_object_track_set_auto
 */

#ifndef VIAME_CORE_READ_OBJECT_TRACK_SET_AUTO_H
#define VIAME_CORE_READ_OBJECT_TRACK_SET_AUTO_H

#include "viame_core_export.h"

#include <vital/algo/read_object_track_set.h>

#include <memory>

namespace viame {

/// \brief Auto-detecting object track set reader.
///
/// Automatically detects the input format based on file extension and content,
/// then delegates to the appropriate specialized reader.
///
/// Supported formats:
///   - DIVE JSON (.dive.json, or .json with tracks/features)
///   - VIAME CSV (.csv)
///
/// Format detection priority:
///   1. Explicit extensions: .dive.json -> DIVE
///   2. General extensions: .csv -> VIAME CSV, .json -> inspect content
///   3. Content inspection for JSON files
///
class VIAME_CORE_EXPORT read_object_track_set_auto
  : public kwiver::vital::algo::read_object_track_set
{
public:
  static constexpr char const* name = "auto";

  static constexpr char const* description =
    "Auto-detecting object track set reader.\n\n"
    "Detects format from file extension and content:\n"
    "  - .dive.json: DIVE JSON format\n"
    "  - .json: Inspects content (looks for tracks/features)\n"
    "  - .csv: VIAME CSV format\n\n"
    "Delegates to appropriate specialized reader.";

  read_object_track_set_auto();
  virtual ~read_object_track_set_auto();

  virtual kwiver::vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void open( std::string const& filename );
  virtual void close();

  virtual bool read_set( kwiver::vital::object_track_set_sptr& set );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_OBJECT_TRACK_SET_AUTO_H
