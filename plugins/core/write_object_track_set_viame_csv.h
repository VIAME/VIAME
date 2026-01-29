/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for write_object_track_set_viame_csv
 */

#ifndef VIAME_CORE_WRITE_OBJECT_TRACK_SET_VIAME_CSV_H
#define VIAME_CORE_WRITE_OBJECT_TRACK_SET_VIAME_CSV_H

#include "viame_core_export.h"

#include <vital/algo/write_object_track_set.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <ctime>
#include <map>
#include <memory>

namespace viame {

class VIAME_CORE_EXPORT write_object_track_set_viame_csv
  : public kwiver::vital::algo::write_object_track_set
{
public:
  PLUGGABLE_IMPL(
    write_object_track_set_viame_csv,
    "Object track set viame_csv writer.\n\n"
    "  - Column(s) 1: Detection or Track ID\n"
    "  - Column(s) 2: Video or Image Identifier\n"
    "  - Column(s) 3: Unique Frame Identifier\n"
    "  - Column(s) 4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    "  - Column(s) 8: Detection Confidence\n"
    "  - Column(s) 9: Target Length (0 or less if uncomputed)\n"
    "  - Column(s) 10-11+: Repeated Species, Confidence Pairs\n",
    PARAM_DEFAULT(
      delimiter, std::string,
      "Delimiter character for output file",
      "," ),
    PARAM_DEFAULT(
      stream_identifier, std::string,
      "Stream identifier to use when no frame UID is available",
      "" ),
    PARAM_DEFAULT(
      model_identifier, std::string,
      "Model identifier to write in metadata header",
      "" ),
    PARAM_DEFAULT(
      version_identifier, std::string,
      "Software version identifier to write in metadata header",
      "" ),
    PARAM_DEFAULT(
      frame_rate, std::string,
      "Frame rate to write in metadata header",
      "" ),
    PARAM_DEFAULT(
      active_writing, bool,
      "When true, write detections as they arrive instead of buffering",
      false ),
    PARAM_DEFAULT(
      write_time_as_uid, bool,
      "Write timestamp as frame UID instead of filename",
      false ),
    PARAM_DEFAULT(
      tot_option, std::string,
      "Track object type option: detection, average, weighted_average, "
      "weighted_average_scaled_by_conf",
      "weighted_average" ),
    PARAM_DEFAULT(
      tot_ignore_class, std::string,
      "Class name to ignore when computing track object type average",
      "" ),
    PARAM_DEFAULT(
      frame_id_adjustment, int,
      "Value to add to frame IDs when writing",
      0 ),
    PARAM_DEFAULT(
      top_n_classes, unsigned,
      "Maximum number of class labels to output (0 for all)",
      0 ),
    PARAM_DEFAULT(
      mask_to_poly_tol, double,
      "Tolerance for mask to polygon conversion (negative to disable)",
      -1.0 ),
    PARAM_DEFAULT(
      mask_to_poly_points, int,
      "Maximum points for mask to polygon conversion (negative to disable)",
      20 )
  )

  virtual ~write_object_track_set_viame_csv() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void write_set( const kwiver::vital::object_track_set_sptr& set,
                          const kwiver::vital::timestamp& ts,
                          const std::string& file_id );

  virtual void close();

private:
  void initialize() override;

  void set_configuration_internal(
    kwiver::vital::config_block_sptr config ) override;

  std::string format_image_id( const kwiver::vital::object_track_state* ts );
  void write_header_info( std::ostream& stream );
  void write_detection_info( std::ostream& stream,
                             const kwiver::vital::detected_object_sptr& det );

  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  std::map< unsigned, kwiver::vital::track_sptr > m_tracks;
  std::map< unsigned, std::string > m_frame_uids;
  std::time_t m_start_time;
};

} // end namespace

#endif // VIAME_CORE_WRITE_OBJECT_TRACK_SET_VIAME_CSV_H
