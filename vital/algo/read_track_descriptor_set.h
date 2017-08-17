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
 * \brief Interface for read_track_descriptor_set
 */

#ifndef VITAL_READ_TRACK_DESCRIPTOR_SET_H
#define VITAL_READ_TRACK_DESCRIPTOR_SET_H

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
 * @brief Read detected object sets
 *
 * This class is the abstract base class for the detected object set
 * writer.
 *
 * Detection sets from multiple images are stored in a single file
 * with enough information to recreate a unique image identifier,
 * usually the file name, and an associated set of track descriptors.
 */
class VITAL_ALGO_EXPORT read_track_descriptor_set
  : public kwiver::vital::algorithm_def<read_track_descriptor_set>
{
public:
  virtual ~read_track_descriptor_set();

  /// Return the name of this algorithm
  static std::string static_type_name() { return "read_track_descriptor_set"; }

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
   *
   * \throws kwiver::vital::file_not_found_exception
   */
  void open( std::string const& filename );

  /// Read track descriptors from an existing stream
  /**
   * This method specifies the input stream to use for reading
   * track descriptors. Using a stream is handy when the track descriptors are
   * available in a stream format.
   *
   * @param strm input stream to use
   */
  void use_stream( std::istream* strm );

  /// Close track descriptor set file.
  /**
   * The currently open track descriptor set file is closed. If there is no
   * currently open file, then this method does nothing.
   */
  void close();

  /// Read next detected object set
  /**
   * This method reads the next set of detected objects from the
   * file. \b False is returned when the end of file is reached.
   *
   * \param[out] set Pointer to the new set of track descriptors. Set may be
   * empty if there are no track descriptors on an image.
   *
   * @return \b true if track descriptors are returned, \b false if end of file.
   */
  virtual bool read_set( kwiver::vital::track_descriptor_set_sptr& set ) = 0;

  /// Determine if input file is at end of file.
  /**
   * This method reports the end of file status for a file open for reading.
   *
   * @return \b true if file is at end.
   */
  bool at_eof() const;

protected:
  read_track_descriptor_set();

  std::istream& stream();

  // Called when a new stream is specified. Allows derived classes to reinitialize.
  virtual void new_stream();

private:
  std::istream* m_stream;
  bool m_stream_owned;
};


/// Shared pointer type for generic read_track_descriptor_set definition type.
typedef std::shared_ptr<read_track_descriptor_set> read_track_descriptor_set_sptr;

} } } // end namespace

#endif // VITAL_READ_TRACK_DESCRIPTOR_SET_H
