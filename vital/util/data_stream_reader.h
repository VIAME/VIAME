// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_UTIL_STREAM_DATA_READER_H
#define VITAL_UTIL_STREAM_DATA_READER_H

#include <vital/util/vital_util_export.h>

#include <vital/util/string_editor.h>

#include <string>
#include <vector>
#include <istream>

namespace kwiver {
namespace vital {

class string_editor;

// ----------------------------------------------------------------
/**
 * @brief Stream reader class
 *
 * This class represents a line based input stream, that removes shell
 * type comments, blank lines and trims trailing white space.
 *
 * The intent of this class is to provide a generally useful reader
 * that provides all common content handling for data files with
 * comments and structural spacing. Using this class to handle the
 * first level of input processing unifies the comment and blank line
 * processing.
 *
 * Example:
\code
  std::ifstream ifs( filename.c_str() );
  if ( ! ifs )
  {
    LOG_ERROR( logger, "Could not open file \"" << filename << "\"" );
    return;
  }

  // use data stream reader to allow for comments and blank lines
  kwiver::vital::data_stream_reader dsr( ifs );

  std::string line;
  while ( dsr.getline( line ) ) // fails on EOF
  {
    // process line

     if (error_found)
     {
       LOG_ERROR( logger, "Error in file at line: " << dsr.line_number() );
     }

  }
\endcode
 */
class VITAL_UTIL_EXPORT data_stream_reader
{
public:

  /**
   * @brief Create new reader on input stream.
   *
   * The supplied stream is used to source the data. It is up to the
   * original owner to close the stream object
   *
   * @param strm Input stream to manage.
   */
  data_stream_reader( std::istream& strm );

  virtual ~data_stream_reader();

  /**
   * @brief Return error status of stream
   *
   * This method returns the error status of the managed stream.
   *
   * @return Returns \b true if there is an error on the stream.
   */
  bool operator!();

  /**
   * @brief Read a line from stream
   *
   *
   * This method returns the next acceptable line from the input
   * stream. There may be some lines that are absorbed by the string
   * editor operations which will not be returned by this method. They
   * will be counted though.
   *
   * @param[out] str String from file is returned here.
   *
   * @return \b true of data is returned. \b false if no data returned (eof)
   */
  bool getline( std::string& str );

  /**
   * @brief Return current line number.
   *
   * This method returns the current line number. Counting started
   * when the stream is opened or the counter is reset. The line
   * number is useful for generating diagnostics.
   *
   * @return Number of line last returned.
   */
  size_t line_number() const;

  /**
   * @brief Reset current line counter.
   *
   * This method resets the current line count. This is not normally
   * needed, but there are times... The count specified how many lines
   * have been returned up to the current point.
   *
   * @param num New current line count.
   */
  void reset_line_number( int num = 0 );

  /**
   * @brief Add additional string processing operations to this reader.
   *
   * This method adds the specified operation to the end of the
   * current list of operations.
   *
   * @param op Pointer to new string editing operation.
   */
  void add_editor( string_edit_operation* op );

private:
  std::istream& m_in_stream;
  int m_line_count;
  string_editor m_string_editor;
}; // end class file_reader

} } // end namespace

#endif // VITAL_UTIL_STREAM_DATA_READER_H
