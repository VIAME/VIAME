// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief The regular expression VIAME's parsers use.
///
/// Four patterns in three files, all of them reading configuration: the
/// pipeline tokeniser, the `$TYPE{name}` token expander, and the box
/// drawer's colour specification. They were `kwiversys::RegularExpression`,
/// whose shape this keeps -- `find` then `match( n )` -- so that P8-T05's
/// change of engine is not also a change of call site.

#ifndef KWIVER_VITAL_UTIL_REGEX_H
#define KWIVER_VITAL_UTIL_REGEX_H

#include <viame/algorithm_framework/util/vital_util_export.h>

#include <memory>
#include <string>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// A compiled pattern, and the last thing it matched.
///
/// Search with `find`, then read the groups with `match`. The object holds
/// the result, which is why it is not const: two searches with one object
/// is two answers, and only the last one is readable.
class VITAL_UTIL_EXPORT regex
{
public:
  /// @brief Compile a pattern.
  ///
  /// @throws std::regex_error if the pattern is not one.
  explicit regex( std::string const& pattern );

  ~regex();

  regex( regex const& ) = delete;
  regex& operator=( regex const& ) = delete;

  /// @brief Look for the pattern anywhere in `subject`.
  ///
  /// Anchored only where the pattern says so: a pattern starting `^` matches
  /// at the beginning, and one that does not matches anywhere.
  ///
  /// @return Whether it was found. The groups are only meaningful if so.
  bool find( std::string const& subject );

  /// @brief What group `index` matched in the last successful `find`.
  ///
  /// Group 0 is the whole match. A group that did not participate is the
  /// empty string rather than an error, which is what the token expander
  /// relies on for its optional name.
  std::string match( size_t index ) const;

  /// @brief Where group `index` began, as an offset into the subject.
  ///
  /// Group 0 is the whole match. The token expander uses this and `end` to
  /// copy the text on either side of what it is replacing.
  ///
  /// @return The offset, or 0 for a group that did not match.
  size_t start( size_t index = 0 ) const;

  /// @brief Where group `index` ended, as a one-past-the-end offset.
  size_t end( size_t index = 0 ) const;

private:
  class impl;

  std::unique_ptr< impl > m_impl;
};

} // namespace vital

} // namespace kwiver

#endif // KWIVER_VITAL_UTIL_REGEX_H
