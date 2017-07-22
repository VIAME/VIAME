/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief Interface for detected_object_set output
 */

#ifndef _VITAL_DETECTED_OBJECT_SET_OUTPUT_H
#define _VITAL_DETECTED_OBJECT_SET_OUTPUT_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>

#include <vital/types/detected_object_set.h>

#include <string>
#include <fstream>

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------
/**
 * @brief Read and write detected object sets
 *
 * This class is the abstract base class for the detected object set
 * reader and writer.
 *
 * Detection sets from multiple images are stored in a single file
 * with enough information to recreate a unique image identifier,
 * usually the file name, and an associated wet of detections.
 *
 */
class VITAL_ALGO_EXPORT detected_object_set_output
  : public kwiver::vital::algorithm_def<detected_object_set_output>
{
public:
  virtual ~detected_object_set_output();

  /// Return the name of this algorithm
  static std::string static_type_name() { return "detected_object_set_output"; }

  /// Open a file of detection sets.
  /**
   * This method opens a detection set file for writing.
   *
   * \param filename Name of file to open
   *
   * \throws kwiver::vital::path_not_exists Thrown when the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_file Thrown when the given path does
   *    not point to a file (i.e. it points to a directory).
   */
  void open( std::string const& filename );

  /// Write detections to an existing stream
  /**
   * This method specifies the output stream to use for writing
   * detections. Using a stream is handy when the detections output is
   * available in a stream format.
   *
   * @param strm output stream to use
   */
  void use_stream( std::ostream* strm );

  /// Close detection set file.
  /**
   * The currently open detection set file is closed. If there is no
   * currently open file, then this method does nothing.
   */
  void close();

  /// Write detected object set.
  /**
   * This method writes the specified detected object set and image
   * name to the currently open file.
   *
   * \param set Detected object set
   * \param image_path File path to image associated with the detections.
   */
  virtual void write_set( const kwiver::vital::detected_object_set_sptr set,
                          std::string const& image_path ) = 0;


protected:
  detected_object_set_output();

  std::ostream& stream();
  std::string const& filename();

private:
  std::ostream* m_stream;
  bool m_stream_owned;

  std::string m_filename;
};


/// Shared pointer type for generic detected_object_set_output definition type.
typedef std::shared_ptr<detected_object_set_output> detected_object_set_output_sptr;

} } } // end namespace

#endif // _VITAL_DETECTED_OBJECT_SET_OUTPUT_H
