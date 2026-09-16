/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Object track set writer producing DIVE JSON (version 2)
 */

#ifndef VIAME_FILE_IO_WRITE_OBJECT_TRACK_SET_DIVE_H
#define VIAME_FILE_IO_WRITE_OBJECT_TRACK_SET_DIVE_H

#include "viame_file_io_export.h"

#include <viame/algorithm_framework/algo/write_object_track_set.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

#include <iosfwd>
#include <memory>
#include <vector>

namespace viame {

/// Serialization settings shared by the track and detection DIVE writers
struct VIAME_FILE_IO_EXPORT dive_write_options
{
  int frame_id_adjustment = 0;
  unsigned top_n_classes = 0;
  bool pretty_print = true;
};

/// Write tracks as a DIVE JSON annotation document.
///
/// Each track becomes one DIVE track keyed by its id, with one feature per
/// state carrying the box, head/tail keypoints, polygon geometry, length,
/// notes and "(atr)" style detection attributes. Track confidence pairs are
/// the class scores averaged over the states that carry a classification.
VIAME_FILE_IO_EXPORT
void write_dive_json( std::ostream& stream,
                      std::vector< viame::track_sptr > const& tracks,
                      dive_write_options const& options );

// -----------------------------------------------------------------------------
class VIAME_FILE_IO_EXPORT write_object_track_set_dive
  : public viame::algo::write_object_track_set
{
public:
  PLUGGABLE_IMPL_NAMED(
    write_object_track_set_dive, "dive",
    "Object track set writer producing DIVE JSON (version 2).\n\n"
    "Tracks are keyed by id with one feature per frame holding the box "
    "bounds, keyframe flag, optional head/tail points, polygon geometry, "
    "fish length, notes and per-detection attributes. Class labels are "
    "written as track confidence pairs. See "
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

  virtual ~write_object_track_set_dive() = default;

  virtual bool check_configuration( viame::config_block_sptr config ) const;

  virtual void write_set( const viame::object_track_set_sptr& set,
                          const viame::timestamp& ts,
                          const std::string& file_id );

  virtual void close();

private:
  void initialize() override;

  std::map< viame::track_id_t, viame::track_sptr > m_tracks;
};

} // end namespace

#endif // VIAME_FILE_IO_WRITE_OBJECT_TRACK_SET_DIVE_H
