// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_UTIL_STRING_FORMAT_H
#define KWIVER_VITAL_UTIL_STRING_FORMAT_H

#include <vital/util/vital_util_export.h>

#include <stdarg.h>  // For va_start, etc.
#include <string>
#include <vector>
#include <set>

namespace kwiver {
namespace vital {

/**
 * @brief Printf style formatting for std::string
 *
 * This function creates a std::string from an printf style input
 * format specifier and a list of values.
 *
 * @param fmt_str Formatting string using embedded printf format specifiers.
 *
 * @return Formatted string.
 */
VITAL_UTIL_EXPORT std::string
string_format( const std::string fmt_str, ... );

/**
 * @brief Does string start with pattern
 *
 * This function checks to see if the input starts with the supplied
 * pattern.
 *
 * @param input String to be checked
 * @param pattern String to use for checking.
 *
 * @return \b true if string starts with pattern
 */
inline bool
starts_with( const std::string& input, const std::string& pattern)
{
  return (0 == input.compare( 0, pattern.size(), pattern ) );
}

//@}
/**
 * @brief Join a set of strings with specified separator.
 *
 * A single string is created and returned from the supplied vector of
 * strings with the specified separator inserted between
 * strings. There is no trailing separator.
 *
 * @param elements Container of elements to join
 * @param str_separator String to be placed between elements
 *
 * @return Single string with all elements joined with separator.
 */
VITAL_UTIL_EXPORT std::string
join( const std::vector<std::string>& elements, const std::string& str_separator);

VITAL_UTIL_EXPORT std::string
join( const std::set<std::string>& elements, const std::string& str_separator);
//@}

/**
 * @brief Removes duplicate strings while preserving original order.
 *
 * Modifies a vector of strings inplace by removing duplicates encountered in a
 * forward iteration. The result is a unique vector of strings that preserves
 * the forwards order.
 *
 * @param[in,out] items Vector of strings to modify inplace
 */
VITAL_UTIL_EXPORT void
erase_duplicates(std::vector<std::string>& items);

/**
 * @brief Removes whitespace from left side of string.
 *
 * @param[in,out] s String to be trimmed in place.
 * @return Modified string
 */
inline std::string&
left_trim( std::string& s )
{
  s.erase(  0, s.find_first_not_of( " \t\n\r\f\v" ) );
  return s;
}

/**
 * @brief Removes whitespace from right size of string.
 *
 * @param[in,out] s String to be trimmed in place
 * @return Modified string
 */
inline std::string&
right_trim( std::string& s )
{
  s.erase( s.find_last_not_of( " \t\n\r\f\v" ) + 1 );
  return s;
}

/**
 * @brief Removes whitespace from both ends of a string.
 *
 * @param[in,out] s String to be trimmed in place
 * @return Modified string
 */
inline std::string&
string_trim( std::string& s )
{
  right_trim(s);
  left_trim(s);
  return s;
}

} } // end namespace

#endif /* KWIVER_VITAL_UTIL_STRING_FORMAT_H */
