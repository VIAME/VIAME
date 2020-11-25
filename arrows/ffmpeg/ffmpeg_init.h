// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef FFMPEG_INIT_H_
#define FFMPEG_INIT_H_

// ---------------------------------------------------------------------------
/**
* @brief Initialize the ffmpeg codecs.
*
* This will be called by the ffmpeg streams, so you shouldn't need to worry
* about it. This function can be called multiple times, but the real ffmpeg
* initialization routine will run only once.
*/
void ffmpeg_init();

#endif // FFMPEG_INIT_H_

