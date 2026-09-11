/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Detected object set writer producing DIVE JSON (version 2)
 */

#ifndef VIAME_CORE_WRITE_DETECTED_OBJECT_SET_DIVE_H
#define VIAME_CORE_WRITE_DETECTED_OBJECT_SET_DIVE_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_output.h>
#include <vital/plugin_management/pluggable_macro_magic.h>
#include <vital/types/track.h>

#include <vector>

namespace viame {

// -----------------------------------------------------------------------------
class VIAME_CORE_EXPORT write_detected_object_set_dive
  : public kwiver::vital::algo::detected_object_set_output
{
public:
  PLUGGABLE_IMPL_NAMED(
    write_detected_object_set_dive, "dive",
    "Detected object set writer producing DIVE JSON (version 2).\n\n"
    "Every detection becomes a single-frame DIVE track. Frames are numbered "
    "by the order in which sets are written, starting at zero. See "
    "https://kitware.github.io/dive/DataFormats/",
    PARAM_DEFAULT(
      frame_id_adjustment, int,
      "Value to add to frame IDs when writing",
      0 ),
    PARAM_DEFAULT(
      top_n_classes, unsigned,
      "Maximum number of class labels to output (0 for all)",
      0 ),
    PARAM_DEFAULT(
      pretty_print, bool,
      "Indent the output document for readability",
      true )
  )

  virtual ~write_detected_object_set_dive() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void write_set( const kwiver::vital::detected_object_set_sptr set,
                          std::string const& image_name );

  virtual void close();

private:
  void initialize() override;

  int m_frame_number;
  kwiver::vital::track_id_t m_next_id;
  std::vector< kwiver::vital::track_sptr > m_tracks;
};

} // end namespace

#endif // VIAME_CORE_WRITE_DETECTED_OBJECT_SET_DIVE_H
