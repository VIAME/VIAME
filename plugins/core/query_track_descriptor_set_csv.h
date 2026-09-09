/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for query_track_descriptor_set_csv
 */

#ifndef VIAME_CORE_QUERY_TRACK_DESCRIPTOR_SET_CSV_H
#define VIAME_CORE_QUERY_TRACK_DESCRIPTOR_SET_CSV_H

#include "viame_core_export.h"

#include <vital/algo/query_track_descriptor_set.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

/**
 * @brief Query track descriptors from per-video files in an index folder
 *
 * The file-backed search index keeps one set of files per indexed video
 * (stream), all sharing a basename inside one folder:
 *
 *   <name>.index             marker / manifest; every basename with one is loaded
 *   <name>_descriptors.csv   track descriptors (kwiver csv: uid, type, track
 *                            references, optional raw vector, history)
 *   <name>_tracks.csv        object tracks (viame_csv) referenced by the descriptors
 *
 * Every file is read once, on the first lookup, into a uid -> (video name,
 * descriptor, tracks) map, mirroring what the database-backed
 * implementation answers from its tables.
 */
class VIAME_CORE_EXPORT query_track_descriptor_set_csv
  : public kwiver::vital::algo::query_track_descriptor_set
{
public:
  PLUGGABLE_IMPL_NAMED(
    query_track_descriptor_set_csv, "csv",
    "Queries track descriptors from per-video CSV files in an index folder.",
    PARAM_DEFAULT(
      database_folder, std::string,
      "Folder containing the per-video index files.",
      "" ),
    PARAM_DEFAULT(
      index_postfix, std::string,
      "Postfix of the marker file that identifies an indexed video basename.",
      ".index" ),
    PARAM_DEFAULT(
      descriptor_postfix, std::string,
      "Postfix added to a basename for its track descriptor file.",
      "_descriptors.csv" ),
    PARAM_DEFAULT(
      track_postfix, std::string,
      "Postfix added to a basename for its object track file.",
      "_tracks.csv" ),
    PARAM_DEFAULT(
      descriptor_reader_type, std::string,
      "read_track_descriptor_set implementation used for descriptor files.",
      "csv" ),
    PARAM_DEFAULT(
      track_reader_type, std::string,
      "read_object_track_set implementation used for track files.",
      "viame_csv" ) )

  virtual ~query_track_descriptor_set_csv();

  bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

  bool get_track_descriptor( std::string const& uid, desc_tuple_t& result ) override;

  void use_tracks_for_history( bool value ) override;

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_CORE_QUERY_TRACK_DESCRIPTOR_SET_CSV_H
