/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Interface for track_descriptor_set output
 */

#ifndef VITAL_WRITE_TRACK_DESCRIPTOR_SET_H
#define VITAL_WRITE_TRACK_DESCRIPTOR_SET_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>

#include <vital/types/track_descriptor_set.h>

#include <string>
#include <fstream>

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------------------------
/**
 * @brief Read and write detected object sets
 *
 * This class is the abstract base class for the detected object set
 * reader and writer.
 *
 * Detection sets from multiple images are stored in a single file
 * with enough information to recreate a unique image identifier,
 * usually the file name, and an associated wet of track descriptors.
 *
 */
class VITAL_ALGO_EXPORT write_track_descriptor_set
  : public kwiver::vital::algorithm_def<write_track_descriptor_set>
{
public:
  virtual ~write_track_descriptor_set();

  /// Return the name of this algorithm
  static std::string static_type_name() { return "write_track_descriptor_set"; }

  /// Open a file of track descriptor sets.
  /**
   * This method opens a track descriptor set file for reading.
   *
   * \param filename Name of file to open
   *
   * \throws kwiver::vital::path_not_exists Thrown when the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_file Thrown when the given path does
   *    not point to a file (i.e. it points to a directory).
   */
  void open( std::string const& filename );

  /// Write track descriptors to an existing stream
  /**
   * This method specifies the output stream to use for reading
   * track descriptors. Using a stream is handy when the track descriptors are
   * available in a stream format.
   *
   * @param strm output stream to use
   */
  void use_stream( std::ostream* strm );

  /// Close track descriptor set file.
  /**
   * The currently open track descriptor set file is closed. If there is no
   * currently open file, then this method does nothing.
   */
  void close();

  /// Write detected object set.
  /**
   * This method writes the specified detected object to file.
   *
   * \param set Detected object set
   */
  virtual void write_set( const kwiver::vital::track_descriptor_set_sptr set ) = 0;


protected:
  write_track_descriptor_set();

  std::ostream& stream();
  std::string const& filename();

private:
  std::ostream* m_stream;
  bool m_stream_owned;

  std::string m_filename;
};


/// Shared pointer type for generic write_track_descriptor_set definition type.
typedef std::shared_ptr<write_track_descriptor_set> write_track_descriptor_set_sptr;

} } } // end namespace

#endif // VITAL_WRITE_TRACK_DESCRIPTOR_SET_H
