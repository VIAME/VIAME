// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/util/regex.h>

#include <regex>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
class regex::impl
{
public:
  explicit impl( std::string const& pattern )
    : m_pattern( pattern )
  {}

  // The subject is kept because `std::smatch` holds iterators into it; the
  // caller's string may be a temporary, and `match()` is read afterwards.
  std::regex m_pattern;
  std::string m_subject;
  std::smatch m_match;
};

// ----------------------------------------------------------------------------
regex
::regex( std::string const& pattern )
  : m_impl( new regex::impl( pattern ) )
{}

regex
::~regex() = default;

// ----------------------------------------------------------------------------
bool
regex
::find( std::string const& subject )
{
  m_impl->m_subject = subject;

  return std::regex_search(
    m_impl->m_subject, m_impl->m_match, m_impl->m_pattern );
}

// ----------------------------------------------------------------------------
std::string
regex
::match( size_t index ) const
{
  if( index >= m_impl->m_match.size() )
  {
    return {};
  }

  // An optional group that did not take part is empty rather than absent,
  // which is the answer `$ENV{}` needs.
  return m_impl->m_match[ index ].matched
         ? m_impl->m_match[ index ].str()
         : std::string{};
}

// ----------------------------------------------------------------------------
size_t
regex
::start( size_t index ) const
{
  if( index >= m_impl->m_match.size() || !m_impl->m_match[ index ].matched )
  {
    return 0;
  }

  return static_cast< size_t >( m_impl->m_match.position( index ) );
}

// ----------------------------------------------------------------------------
size_t
regex
::end( size_t index ) const
{
  if( index >= m_impl->m_match.size() || !m_impl->m_match[ index ].matched )
  {
    return 0;
  }

  return static_cast< size_t >(
    m_impl->m_match.position( index ) + m_impl->m_match.length( index ) );
}

} // namespace vital

} // namespace kwiver
