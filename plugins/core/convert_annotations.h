/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Annotation file conversion through the registered vital readers
 *        and writers, without a pipeline.
 */

#ifndef VIAME_CORE_CONVERT_ANNOTATIONS_H
#define VIAME_CORE_CONVERT_ANNOTATIONS_H

#include "viame_core_export.h"

#include <vital/logger/logger.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace viame {

/// Settings for one annotation file conversion
struct VIAME_CORE_EXPORT annotation_conversion_options
{
  /// Reader implementation name, or "auto" to detect from the file
  std::string input_format = "auto";

  /// Writer implementation name (required)
  std::string output_format;

  /// Where frames come from: an image folder, an image list file, or a
  /// video. Empty means the frames are taken from the annotation file alone.
  std::string frame_source;

  /// Frames per second used for timestamps and, for videos, for numbering
  /// the output frames at a reduced rate. Zero keeps the native rate.
  double frame_rate = 0.0;

  /// Added to every output frame number, when the writer supports it
  int frame_offset = 0;

  /// Extra "key=value" configuration applied to the reader and writer
  std::vector< std::pair< std::string, std::string > > reader_settings;
  std::vector< std::pair< std::string, std::string > > writer_settings;
};

/// What a conversion did, for reporting
struct VIAME_CORE_EXPORT annotation_conversion_summary
{
  std::string reader;
  std::string writer;
  std::string frame_source;
  bool used_tracks = false;
  std::size_t frames = 0;
  std::size_t tracks = 0;
  std::size_t detections = 0;
};

/// Image and video extensions (lower case, with the leading dot) used to
/// recognise data sitting alongside annotation files
VIAME_CORE_EXPORT std::vector< std::string > const& default_image_extensions();
VIAME_CORE_EXPORT std::vector< std::string > const& default_video_extensions();

/// Reader implementation for an annotation file judged from its extension
/// and, for JSON and XML, its content. Empty when the file is not a
/// recognised annotation file.
VIAME_CORE_EXPORT std::string detect_annotation_format( std::string const& path );

/// Reader or writer implementation implied by a file extension such as
/// ".csv" or a path ending in one; empty when unknown
VIAME_CORE_EXPORT std::string format_from_extension( std::string const& path_or_ext );

/// Customary file extension (with the dot) for a format name
VIAME_CORE_EXPORT std::string extension_for_format( std::string const& format );

/// Annotation files under a folder, searched recursively and sorted. When a
/// format is given only files with its customary extension are returned;
/// otherwise every file that detect_annotation_format() recognises.
VIAME_CORE_EXPORT std::vector< std::string > list_annotation_files(
  std::string const& folder, std::string const& format = "" );

/// Imagery sitting next to an annotation file: the folder itself when it
/// holds images, or a video in it (one sharing the annotation's name is
/// preferred). Empty when no data is found or the choice is ambiguous.
VIAME_CORE_EXPORT std::string find_frame_source_alongside(
  std::string const& annotation_path );

/// Convert one annotation file. Tracks are carried when both sides support
/// them, otherwise per-frame detections. Returns false on failure, with the
/// reason logged.
VIAME_CORE_EXPORT bool convert_annotation_file(
  std::string const& input_path,
  std::string const& output_path,
  annotation_conversion_options const& options,
  annotation_conversion_summary& summary,
  kwiver::vital::logger_handle_t logger );

} // end namespace viame

#endif // VIAME_CORE_CONVERT_ANNOTATIONS_H
