/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Read and write the OpenCV FileStorage subset VIAME uses
///
/// `cv::FileStorage` is the only reason several calibration files exist in
/// this format, and P7-T05 replaces it. What is here is the subset VIAME
/// writes and reads, which `design/lite-removals.md` section 2.5 lists:
///
/// * YAML, read and written -- `%YAML:1.0`, block maps, block and flow
///   sequences, scalars, and `!!opencv-matrix` with rows, cols, dt and data;
/// * XML, read only -- `<opencv_storage>`, nested elements, and `<_>` as a
///   sequence entry, which is what `Model_SVM.xml` is.
///
/// JSON, which FileStorage also supports, is not here: nothing in VIAME
/// writes an OpenCV JSON document, and the JSON it does read is its own
/// shape and has its own reader in `camera_rig_io`.
///
/// The writer reproduces OpenCV's layout exactly, down to where a long data
/// array wraps, so a regenerated calibration file diffs cleanly against one
/// FileStorage wrote. `tests/library/file_io` checks that byte for byte.

#ifndef VIAME_FILE_IO_OPENCV_YAML_H
#define VIAME_FILE_IO_OPENCV_YAML_H

#include "viame_file_io_export.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace viame {

namespace file_io {

// ----------------------------------------------------------------------------
/// What a document could not be read as.
class VIAME_FILE_IO_EXPORT parse_error : public std::runtime_error
{
public:
  parse_error( std::string const& what ) : std::runtime_error( what ) {}
};

// ----------------------------------------------------------------------------
/// One node of a FileStorage document.
///
/// A tree rather than a flattened path map, because the shape is half of
/// what a reader has to get right: a map stays a map and a sequence stays a
/// sequence. A matrix is a map of `rows`, `cols`, `dt` and `data` carrying a
/// flag, which is what the format actually holds -- decoding it to an array
/// here would lose `dt`, and `dt` is what says whether the data is integers
/// or doubles.
class VIAME_FILE_IO_EXPORT node
{
public:
  enum class kind
  {
    none,
    integer,
    real,
    string,
    sequence,
    map,
  };

  /// An ordered key and value, since a document's order is part of it.
  typedef std::pair< std::string, node > entry;

  node();
  explicit node( long long value );
  explicit node( double value );
  explicit node( std::string value );

  static node sequence_of( std::vector< node > values );
  static node map_of( std::vector< entry > entries, bool matrix = false );

  /// A `!!opencv-matrix` of \p rows by \p cols of type \p dt.
  ///
  /// \p dt is OpenCV's one letter type code: `u` unsigned byte, `c` signed
  /// byte, `w` unsigned short, `s` short, `i` int, `f` float, `d` double.
  static node matrix( int rows, int cols, std::string dt,
                      std::vector< double > const& data );

  kind type() const { return kind_; }

  bool is_none() const { return kind_ == kind::none; }
  bool is_integer() const { return kind_ == kind::integer; }
  bool is_real() const { return kind_ == kind::real; }
  bool is_number() const { return is_integer() || is_real(); }
  bool is_string() const { return kind_ == kind::string; }
  bool is_sequence() const { return kind_ == kind::sequence; }
  bool is_map() const { return kind_ == kind::map; }

  /// Whether this map carried the `!!opencv-matrix` tag.
  bool is_matrix() const { return matrix_; }

  long long as_integer() const;
  double as_double() const;
  std::string const& as_string() const;

  std::vector< node > const& values() const { return values_; }
  std::vector< entry > const& entries() const { return entries_; }

  /// The value under \p key, or a none node.
  node const& operator[]( std::string const& key ) const;
  bool has( std::string const& key ) const;

  /// A matrix's data as doubles, whatever its `dt`.
  ///
  /// Throws unless this is a matrix whose `data` holds exactly
  /// `rows * cols` numbers.
  std::vector< double > matrix_data() const;
  int matrix_rows() const;
  int matrix_cols() const;
  std::string matrix_type() const;

private:
  kind kind_;
  bool matrix_;
  long long integer_;
  double real_;
  std::string string_;
  std::vector< node > values_;
  std::vector< entry > entries_;
};

// ----------------------------------------------------------------------------
/// Parse \p path, choosing YAML or XML by what is in it.
///
/// Returns the root, which is always a map. Throws `parse_error` on anything
/// it cannot read rather than returning a half-filled document: a
/// calibration silently missing its distortion coefficients is worse than a
/// calibration that fails to load.
VIAME_FILE_IO_EXPORT
node read( std::string const& path );

/// Parse YAML text that has already been read.
VIAME_FILE_IO_EXPORT
node parse_yaml( std::string const& text );

/// Parse XML text that has already been read.
VIAME_FILE_IO_EXPORT
node parse_xml( std::string const& text );

// ----------------------------------------------------------------------------
/// Write \p root to \p path as OpenCV YAML.
///
/// \p root must be a map. Throws on an error opening or writing the file.
VIAME_FILE_IO_EXPORT
void write( std::string const& path, node const& root );

/// The YAML text `write` would produce.
VIAME_FILE_IO_EXPORT
std::string to_yaml( node const& root );

// ----------------------------------------------------------------------------
/// How a double is spelled in an OpenCV document.
///
/// `%.16e`, except that a value equal to its own rounding is written as the
/// integer and a full stop -- `0.`, `1.`, `-3.`. That is `icvDoubleToString`,
/// and it is why a calibration file is full of bare `0.`
VIAME_FILE_IO_EXPORT
std::string double_to_string( double value );

/// The same for a float: `%.8e`, with the same integer rule.
VIAME_FILE_IO_EXPORT
std::string float_to_string( float value );

} // namespace file_io

} // namespace viame

#endif // VIAME_FILE_IO_OPENCV_YAML_H
