/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Baseline TIFF, read and written in house
///
/// stb does not do TIFF and never has, and VIAME's data is full of it: the
/// survey imagery and the calibration captures are 8 and 16 bit stripped
/// TIFF, uncompressed, LZW or PackBits. This reads that, which is the subset
/// `design/lite-removals.md` 2.2 scopes, and writes uncompressed strips.
///
/// What it deliberately does not read is a tiled TIFF. Tiles are a separate
/// layout with their own offset tables, nothing VIAME writes produces one,
/// and the one place a tiled file could arrive from -- a third party's
/// imagery -- is exactly where a fallback is the right answer. `can_read`
/// says so before `read` is called, so the caller can hand the file to the
/// python `pil` image_io instead of failing.

#ifndef VIAME_VIDEO_IO_CODECS_TIFF_H
#define VIAME_VIDEO_IO_CODECS_TIFF_H

#include "viame_video_io_export.h"

#include <viame/core_types/image.h>

#include <string>

namespace viame {

namespace codecs {

namespace tiff {

/// @brief Whether these bytes begin a TIFF file.
///
/// The first four: "II" or "MM" for the byte order, then 42 in that order.
VIAME_VIDEO_IO_EXPORT
bool is_tiff( void const* data, size_t size );

/// @brief Why a TIFF cannot be read here, or empty if it can.
///
/// Separate from `read` so that a caller can fall back before failing. The
/// message names the feature, so a log line says which file wanted what.
///
/// @param filename the file to inspect
VIAME_VIDEO_IO_EXPORT
std::string unsupported_reason( std::string const& filename );

/// @brief Read a baseline TIFF.
///
/// Strips only, contiguous planar configuration, 8 or 16 bits per sample,
/// one, three or four samples per pixel, uncompressed, LZW (with or without
/// the horizontal differencing predictor) or PackBits.
///
/// @param filename the file to read
/// @returns the image, 8 or 16 bit to match the file
/// @throws std::runtime_error if the file is not one of those
VIAME_VIDEO_IO_EXPORT
kwiver::vital::image read( std::string const& filename );

/// @brief Write an uncompressed, stripped TIFF.
///
/// One strip, contiguous samples, in the machine's byte order. 8 and 16 bit
/// images are written as they are; anything else is an error rather than a
/// silent conversion, because the caller is the one that knows what losing
/// the range would mean.
///
/// @param filename the file to write
/// @param image the image to write
/// @throws std::runtime_error if the pixel type is not 8 or 16 bit
VIAME_VIDEO_IO_EXPORT
void write( std::string const& filename, kwiver::vital::image const& image );

} // end namespace tiff

} // end namespace codecs

} // end namespace viame

#endif
