// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface for detected_object_set output

#ifndef _VITAL_DETECTED_OBJECT_SET_OUTPUT_H
#define _VITAL_DETECTED_OBJECT_SET_OUTPUT_H

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/vital_config.h>

#include <viame/core_types/detected_object_set.h>

#include <fstream>
#include <string>

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------
/// @brief Read and write detected object sets
///
/// This class is the abstract base class for the detected object set
/// reader and writer.
///
/// Detection sets from multiple images are stored in a single file
/// with enough information to recreate a unique image identifier,
/// usually the file name, and an associated wet of detections.
///
class VITAL_ALGO_EXPORT detected_object_set_output
  : public kwiver::vital::algorithm
{
public:
  virtual ~detected_object_set_output();

  detected_object_set_output();
  PLUGGABLE_INTERFACE_NO_DESTR( detected_object_set_output );
  /// Open a file of detection sets.
  ///
  /// This method opens a detection set file for writing.
  ///
  /// \param filename Name of file to open
  ///
  /// \throws kwiver::vital::path_not_exists Thrown when the given path does not
  /// exist.
  ///
  /// \throws kwiver::vital::path_not_a_file Thrown when the given path does
  ///    not point to a file (i.e. it points to a directory).
  virtual void open( std::string const& filename );

  /// Write detections to an existing stream
  ///
  /// This method specifies the output stream to use for writing
  /// detections. Using a stream is handy when the detections output is
  /// available in a stream format.
  ///
  /// @param strm output stream to use
  void use_stream( std::ostream* strm );

  /// Close detection set file.
  ///
  /// The currently open detection set file is closed. If there is no
  /// currently open file, then this method does nothing.
  virtual void close();

  /// Write detected object set.
  ///
  /// This method writes the specified detected object set and image
  /// name to the currently open file.
  ///
  /// \param set Detected object set
  /// \param image_path File path to image associated with the detections.
  virtual void write_set(
    const kwiver::vital::detected_object_set_sptr set,
    std::string const& image_path ) = 0;

  /// Perform end-of-stream actions.
  ///
  /// This method writes any necessary final data to the currently open file.
  virtual void complete() {}

  ///@{
  /// Filename property
  /// @note  Required for accessing it as a python property
  std::string
  get_filename() const { return m_filename; }
  void set_filename( std::string filename ) { m_filename = filename; }
  ///@}

protected:
  std::ostream& stream();

  std::string const& filename();

private:
  std::ostream* m_stream;
  bool m_stream_owned;

  std::string m_filename;
};

/// Shared pointer type for generic detected_object_set_output definition type.
typedef std::shared_ptr< detected_object_set_output >
  detected_object_set_output_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // _VITAL_DETECTED_OBJECT_SET_OUTPUT_H
