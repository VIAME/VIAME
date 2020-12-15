// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for write_object_track_set
 */

#ifndef VITAL_WRITE_OBJECT_TRACK_SET_H
#define VITAL_WRITE_OBJECT_TRACK_SET_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>

#include <vital/types/object_track_set.h>

#include <string>
#include <fstream>

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------------------------
/**
 * @brief Read and write object track sets
 *
 * This class is the abstract base class for the object track set writer.
 *
 * Track sets from multiple images are stored in a single file with
 * enough information to recreate a unique image identifier, usually a frame
 * number, and an associated wet of object tracks.
 */
class VITAL_ALGO_EXPORT write_object_track_set
  : public kwiver::vital::algorithm_def<write_object_track_set>
{
public:
  virtual ~write_object_track_set();

  /// Return the name of this algorithm
  static std::string static_type_name() { return "write_object_track_set"; }

  /// Open a file of object track sets.
  /**
   * This method opens a object track set file for reading.
   *
   * \param filename Name of file to open
   *
   * \throws kwiver::vital::path_not_exists Thrown when the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_file Thrown when the given path does
   *    not point to a file (i.e. it points to a directory).
   */
  virtual void open( std::string const& filename );

  /// Write object tracks to an existing stream
  /**
   * This method specifies the output stream to use for writing
   * object tracks.
   *
   * @param strm output stream to use
   */
  virtual void use_stream( std::ostream* strm );

  /// Close object track set file.
  /**
   * The currently open object track set file is closed. If there is no
   * currently open file, then this method does nothing.
   */
  virtual void close();

  /// Write object track set.
  /**
   * This method writes the specified object track set and image
   * name to the currently open file.
   *
   * \param set Track object set
   * \param ts Timestamp for the current frame
   * \param frame_identifier Identifier for the current frame (e.g. file name)
   */
  virtual void write_set(
    kwiver::vital::object_track_set_sptr const& set,
    kwiver::vital::timestamp const& ts = {},
    std::string const& frame_identifier = {} ) = 0;

protected:
  write_object_track_set();

  std::ostream& stream();
  std::string const& filename();

private:
  std::ostream* m_stream;
  bool m_stream_owned;

  std::string m_filename;
};

/// Shared pointer type for generic write_object_track_set definition type.
typedef std::shared_ptr<write_object_track_set> write_object_track_set_sptr;

} } } // end namespace

#endif // VITAL_WRITE_OBJECT_TRACK_SET_H
