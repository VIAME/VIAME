/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_UTILITIES_FILE_H
#define VIAME_CORE_UTILITIES_FILE_H

#include "viame_core_export.h"

#include <map>
#include <string>
#include <vector>
#include <unordered_set>

namespace viame {

// =============================================================================
// Filesystem utilities
// =============================================================================

/// Check if a file exists at the given location
///
/// \param location Path to check
/// \returns true if path exists and is a regular file
VIAME_CORE_EXPORT
bool does_file_exist( const std::string& location );

/// Check if a folder exists at the given location
///
/// \param location Path to check
/// \returns true if path exists and is a directory
VIAME_CORE_EXPORT
bool does_folder_exist( const std::string& location );

/// List all immediate subdirectories in a folder
///
/// \param location Directory to search
/// \param subfolders Output vector of subdirectory paths
/// \returns true if location exists and is a directory
VIAME_CORE_EXPORT
bool list_all_subfolders( const std::string& location,
                          std::vector< std::string >& subfolders );

/// List files in a folder, optionally filtering by extension
///
/// \param location Directory to search
/// \param filepaths Output vector of file paths
/// \param search_subfolders If true, recurse into subdirectories
/// \param extensions Optional list of extensions to filter (e.g., ".jpg", ".png")
/// \returns true if location exists and is a directory
VIAME_CORE_EXPORT
bool list_files_in_folder( std::string location,
                           std::vector< std::string >& filepaths,
                           bool search_subfolders = false,
                           std::vector< std::string > extensions = std::vector< std::string >() );

/// Create a folder and any necessary parent directories
///
/// \param location Path to create
/// \returns true if folder was created, false if it already exists or on error
VIAME_CORE_EXPORT
bool create_folder( const std::string& location );

/// Check if a folder contains fewer than n files
///
/// \param folder Directory to check
/// \param n Threshold count
/// \returns true if folder contains fewer than n files
VIAME_CORE_EXPORT
bool folder_contains_less_than_n_files( const std::string& folder, unsigned n );

// =============================================================================
// Path manipulation utilities
// =============================================================================

/// Join two path components with a forward slash
///
/// \param p1 First path component
/// \param p2 Second path component
/// \returns Combined path
VIAME_CORE_EXPORT
std::string append_path( const std::string& p1, const std::string& p2 );

/// Get the filename from a path (without directory)
///
/// \param path Full path
/// \returns Filename component only
VIAME_CORE_EXPORT
std::string get_filename_no_path( const std::string& path );

/// Get the filename with its immediate parent directory
///
/// \param path Full path
/// \returns "parent/filename" string
VIAME_CORE_EXPORT
std::string get_filename_with_last_path( const std::string& path );

/// Replace the extension of a filename
///
/// \param file_name Original filename
/// \param ext New extension (including dot, e.g., ".png")
/// \returns Filename with new extension
VIAME_CORE_EXPORT
std::string replace_ext_with( const std::string& file_name, const std::string& ext );

/// Add an extension to a path, handling trailing slashes
///
/// \param path Original path
/// \param ext Extension to add (including dot)
/// \returns Path with extension added
VIAME_CORE_EXPORT
std::string add_ext_unto( const std::string& path, const std::string& ext );

/// Add an auxiliary suffix to a filename before the extension
///
/// \param file_name Original filename
/// \param id Auxiliary ID (0 or 1 adds "_aux", >1 adds "_auxN")
/// \returns Filename with auxiliary suffix
VIAME_CORE_EXPORT
std::string add_aux_ext( const std::string& file_name, unsigned id );

/// Check if a string ends with a given extension
///
/// \param str String to check
/// \param ext Extension to look for
/// \returns true if str ends with ext
VIAME_CORE_EXPORT
bool ends_with_extension( const std::string& str, const std::string& ext );

/// Check if a string ends with any of the given extensions
///
/// \param str String to check
/// \param exts List of extensions to check
/// \returns true if str ends with any extension in exts
VIAME_CORE_EXPORT
bool ends_with_extension( const std::string& str,
                          const std::vector< std::string >& exts );

/// Wrap a string in double quotes
///
/// \param str String to quote
/// \returns Quoted string
VIAME_CORE_EXPORT
std::string add_quotes( const std::string& str );

// =============================================================================
// String parsing utilities
// =============================================================================

/// Trim leading and trailing whitespace from a string
///
/// \param str Input string
/// \returns String with leading/trailing whitespace removed
VIAME_CORE_EXPORT
std::string trim_string( std::string const& str );

/// Trim a line and optionally check if it's a comment
///
/// Removes leading/trailing whitespace from the line. If the trimmed line
/// starts with '#', it is considered a comment.
///
/// \param line Input line to trim
/// \param[out] trimmed Output trimmed string (empty if line is empty or comment)
/// \param skip_comments If true, treat lines starting with '#' as empty
/// \returns true if the line has content (non-empty and not a comment if skip_comments)
VIAME_CORE_EXPORT
bool trim_line( std::string const& line, std::string& trimmed, bool skip_comments = true );

/// Convert string to lowercase
///
/// \param str Input string
/// \returns Lowercase version of string
VIAME_CORE_EXPORT
std::string to_lower( std::string const& str );

/// Check if string ends with suffix (case-insensitive)
///
/// \param str String to check
/// \param suffix Suffix to look for
/// \returns true if str ends with suffix (ignoring case)
VIAME_CORE_EXPORT
bool ends_with_ci( std::string const& str, std::string const& suffix );

/// Split a string into a vector using delimiters
///
/// \param str Input string
/// \param out Output vector of tokens
/// \param delims Delimiter characters (default: whitespace and comma)
VIAME_CORE_EXPORT
void string_to_vector( const std::string& str,
                       std::vector< std::string >& out,
                       const std::string delims = "\n\t\v ," );

/// Split a string into a set using delimiters
///
/// \param str Input string
/// \param out Output set of tokens
/// \param delims Delimiter characters (default: whitespace and comma)
VIAME_CORE_EXPORT
void string_to_set( const std::string& str,
                    std::unordered_set< std::string >& out,
                    const std::string delims = "\n\t\v ," );

// =============================================================================
// File reading utilities
// =============================================================================

/// Read non-empty lines from a file into a vector
///
/// \param fn Filename to read
/// \param out Output vector of lines
/// \param reset If true, clear output vector first (default: true)
/// \returns true on success, false if file cannot be opened
VIAME_CORE_EXPORT
bool file_to_vector( const std::string& fn,
                     std::vector< std::string >& out,
                     bool reset = true );

/// Load all lines from a file into a vector (including empty lines)
///
/// \param file Filename to read
/// \param output Output vector of lines
/// \returns true on success, false if file cannot be opened
VIAME_CORE_EXPORT
bool load_file_list( const std::string& file,
                     std::vector< std::string >& output );

/// Check if a file contains a specific string
///
/// \param file Filename to search
/// \param key String to search for
/// \returns true if key is found in the file
VIAME_CORE_EXPORT
bool file_contains_string( const std::string& file, const std::string& key );

/// Parse frame rate from a file header (looks for "fps:" or "fps=" in first 4 lines)
///
/// \param file Filename to parse
/// \returns Frame rate if found, -1.0 otherwise
VIAME_CORE_EXPORT
double get_file_frame_rate( const std::string& file );

// =============================================================================
// Template processing utilities
// =============================================================================

/// Replace keywords in a template file
///
/// Reads the input file, replaces all occurrences of each keyword with its
/// corresponding value, and writes the result to the output file.
///
/// \param input_file Path to template file
/// \param output_file Path to write result
/// \param replacements Map of keyword->value pairs to replace
/// \returns true on success, false if input file cannot be read or output cannot be written
VIAME_CORE_EXPORT
bool replace_keywords_in_template_file(
    const std::string& input_file,
    const std::string& output_file,
    const std::map< std::string, std::string >& replacements );

/// Copy a file from source to destination
///
/// \param source Path to source file
/// \param destination Path to destination file
/// \returns true on success, false if source cannot be read or destination cannot be written
VIAME_CORE_EXPORT
bool copy_file( const std::string& source, const std::string& destination );

/// Replace keywords in a template and return the result as a string
///
/// \param input_file Path to template file
/// \param replacements Map of keyword->value pairs to replace
/// \param[out] result Output string with replacements applied
/// \returns true on success, false if input file cannot be read
VIAME_CORE_EXPORT
bool replace_keywords_in_template_to_string(
    const std::string& input_file,
    const std::map< std::string, std::string >& replacements,
    std::string& result );

// =============================================================================
// Zip file utilities
// =============================================================================

/// Create a zip file from a list of files
///
/// Each entry in files_to_add maps the filename inside the zip to the source path.
/// Entries where the source path is empty are treated as string content to be
/// written directly (the key is used as the filename, value as content).
///
/// \param zip_path Path for the output zip file
/// \param files_to_add Map of zip entry name -> source file path or content
/// \param string_contents Map of zip entry name -> string content (for non-file data)
/// \returns true on success, false on error
VIAME_CORE_EXPORT
bool create_zip_file(
    const std::string& zip_path,
    const std::map< std::string, std::string >& files_to_add,
    const std::map< std::string, std::string >& string_contents = {} );

} // end namespace viame

#endif /* VIAME_CORE_UTILITIES_FILE_H */
