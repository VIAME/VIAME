// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for read_object_track_set
 */

#ifndef VITAL_READ_OBJECT_TRACK_SET_H
#define VITAL_READ_OBJECT_TRACK_SET_H

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
 * @brief Read object track sets
 *
 * This class is the abstract base class for the object track set writer.
 *
 * Track sets from multiple images are stored in a single file with enough
 * information to recreate a unique image identifier, usually the frame number,
 * and an associated set of object tracks. Alternatively, tracks can be read in
 * batch mode.
 */
class VITAL_ALGO_EXPORT read_object_track_set
  : public kwiver::vital::algorithm_def<read_object_track_set>
{
public:
  virtual ~read_object_track_set();

  /// Return the name of this algorithm
  static std::string static_type_name() { return "read_object_track_set"; }

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
   *
   * \throws kwiver::vital::file_not_found_exception
   */
  virtual void open( std::string const& filename );

  /// Read object tracks from an existing stream
  /**
   * This method specifies the input stream to use for reading
   * object tracks. Using a stream is handy when the object tracks are
   * available in a stream format.
   *
   * @param strm input stream to use
   */
  void use_stream( std::istream* strm );

  /// Close object track set file.
  /**
   * The currently open object track set file is closed. If there is no
   * currently open file, then this method does nothing.
   */
  virtual void close();

  /// Read next object track set
  /**
   * This method reads the next set of track objects from the
   * file. \b False is returned when the end of file is reached.
   *
   * \param[out] set Pointer to the new set of object tracks. Set may be
   * empty if there are no object tracks on an image.
   *
   * @return \b true if object tracks are returned, \b false if end of file.
   */
  virtual bool read_set( kwiver::vital::object_track_set_sptr& set ) = 0;

  /// Determine if input file is at end of file.
  /**
   * This method reports the end of file status for a file open for reading.
   *
   * @return \b true if file is at end.
   */
  bool at_eof() const;

protected:
  read_object_track_set();

  std::istream& stream();

  // Called when a new stream is specified. Allows derived classes to
  // reinitialize.
  virtual void new_stream();

private:
  std::istream* m_stream;
  bool m_stream_owned;
};

/// Shared pointer type for generic read_object_track_set definition type.
typedef std::shared_ptr<read_object_track_set> read_object_track_set_sptr;

} } } // end namespace

#endif // VITAL_READ_OBJECT_TRACK_SET_H
