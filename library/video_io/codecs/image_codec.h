/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Reading and writing the image formats VIAME ships, without OpenCV
///
/// PNG, JPEG and BMP go through the vendored stb headers; TIFF goes through
/// `tiff.h`, which is in house because stb has never done TIFF. Anything else
/// -- and any TIFF outside the baseline subset, a tiled one above all -- is
/// declined here rather than guessed at, so the caller can hand it to the
/// python `pil` image_io. `design/lite-removals.md` 2.2 has the split.

#ifndef VIAME_VIDEO_IO_CODECS_IMAGE_CODEC_H
#define VIAME_VIDEO_IO_CODECS_IMAGE_CODEC_H

#include "viame_video_io_export.h"

#include <viame/core_types/image.h>

#include <string>

namespace viame {

namespace codecs {

/// @brief Whether this file can be read here, and why not when it cannot.
///
/// Asked before `read`, so that a caller can fall back rather than fail. The
/// answer looks at the file, not only at its name: a `.tif` is readable or
/// not depending on how it is laid out inside.
///
/// @param filename the file to inspect
/// @param[out] reason why it cannot be read, when it cannot
/// @returns true when `read` will handle it
VIAME_VIDEO_IO_EXPORT
bool can_read( std::string const& filename, std::string& reason );

/// @brief Whether this image can be written to this file here.
///
/// The extension and the pixels together, because the answer depends on
/// both: PNG, JPEG, BMP and TIFF are the containers, and a 16 bit PNG is the
/// one combination that has to go to the fallback -- stb's PNG encoder takes
/// bytes, and narrowing a 16 bit image on the way out would lose the range
/// without anyone asking for it.
///
/// @param filename the file that would be written
/// @param image the image that would be written
/// @param[out] reason why it cannot be written, when it cannot
VIAME_VIDEO_IO_EXPORT
bool can_write( std::string const& filename,
                kwiver::vital::image const& image, std::string& reason );

/// @brief Read an image.
///
/// The result is 8 or 16 bit to match the file, with one, three or four
/// planes. Channel order is the file's own -- RGB, not BGR: OpenCV's reader
/// handed back BGR and the bridge swapped it on the way into `vital::image`,
/// so what a caller saw was always RGB and still is.
///
/// @param filename the file to read
/// @throws std::runtime_error if the format is one `can_read` declines
VIAME_VIDEO_IO_EXPORT
kwiver::vital::image read( std::string const& filename );

/// @brief Write an image, choosing the encoder by extension.
///
/// JPEG and BMP take 8 bit, and a 16 bit image is saturated on the way out,
/// which is what OpenCV's writer did. A single channel BMP is written as a
/// palettised 8 bit one rather than as replicated 24 bit BGR, so that it
/// reads back as one plane. TIFF takes 8 and 16 bit. PNG takes 8 bit here;
/// 16 bit PNG is `can_write`'s one refusal.
///
/// @param filename the file to write
/// @param image the image to write
/// @throws std::runtime_error if the extension is one `can_write` declines
VIAME_VIDEO_IO_EXPORT
void write( std::string const& filename, kwiver::vital::image const& image );

} // end namespace codecs

} // end namespace viame

#endif
