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
 * \brief Interface for detected_object_set input
 */

#ifndef _VITAL_DETECTED_OBJECT_SET_INPUT_H
#define _VITAL_DETECTED_OBJECT_SET_INPUT_H

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
 * @brief Read detected object sets
 *
 * This class is the abstract base class for the detected object set
 * writer.
 *
 * Detection sets from multiple images are stored in a single file
 * with enough information to recreate a unique image identifier,
 * usually the file name, and an associated set of detections.
 */
class VITAL_ALGO_EXPORT detected_object_set_input
  : public kwiver::vital::algorithm_def<detected_object_set_input>
{
public:
  virtual ~detected_object_set_input();

  /// Return the name of this algorithm
  static std::string static_type_name() { return "detected_object_set_input"; }

  /// Open a file of detection sets.
  /**
   * This method opens a detection set file for reading.
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

  /// Read detections from an existing stream
  /**
   * This method specifies the input stream to use for reading
   * detections. Using a stream is handy when the detections are
   * available in a stream format.
   *
   * @param strm input stream to use
   */
  void use_stream( std::istream* strm );

  /// Close detection set file.
  /**
   * The currently open detection set file is closed. If there is no
   * currently open file, then this method does nothing.
   */
  void close();

  /// Read next detected object set
  /**
   * This method reads the next set of detected objects from the
   * file. \b False is returned when the end of file is reached.
   *
   * \param[out] set Pointer to the new set of detections. Set may be
   * empty if there are no detections on an image.
   *
   * \param[in,out] image_name Name of the image that goes with the
   * detections. This string may be empty depending on the source
   * format. If the read format also contains filenames, the string
   * can be used as an input to get all detections on said frame.
   *
   * @return \b true if detections are returned, \b false if end of file.
   */
  virtual bool read_set( kwiver::vital::detected_object_set_sptr & set, std::string& image_name ) = 0;

  /// Determine if input file is at end of file.
  /**
   * This method reports the end of file status for a file open for reading.
   *
   * @return \b true if file is at end.
   */
  bool at_eof() const;

protected:
  detected_object_set_input();

  std::istream& stream();

  // Called when a new stream is specified. Allows derived classes to
  // reinitialize.
  virtual void new_stream();

private:
  std::istream* m_stream;
  bool m_stream_owned;
};


/// Shared pointer type for generic detected_object_set_input definition type.
typedef std::shared_ptr<detected_object_set_input> detected_object_set_input_sptr;

} } } // end namespace

#endif // _VITAL_DETECTED_OBJECT_SET_INPUT_H
