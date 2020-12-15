// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of the tokenize function.
 */

#ifndef VITAL_TOKENIZE_H
#define VITAL_TOKENIZE_H

#include <string>

namespace kwiver {
namespace vital {

enum {
  TokenizeNoTrimEmpty = 0,
  TokenizeTrimEmpty = 1
};

/**
 * @brief Split string into set of tokens.
 *
 * Split the input string based on the set of supplied characters. The
 * detected tokens are added to the end of the \c tokens container,
 * which means the container can start with some contents and the new
 * tokens will be added.
 *
 * If \c trimEmpty is set to \b false (the default state), consecutive
 * delimiters will create empty tokens in the container.
 *
 * @param[in] str String to split
 * @param[in,out] tokens Container tokens are added to
 * @param[in] delimiters List of delimiters used for splitting
 * @param[in] trimEmpty \b false will add empty tokens to container
 * @tparam ContainerT Any container that supports push_back()
 */
template < class ContainerT >
void
tokenize( std::string const& str, // i: string to tokenize
          ContainerT& tokens,     // o: list of tokens
          std::string const& delimiters = " ",
          bool trimEmpty = false )
{
  std::string::size_type pos, lastPos = 0;

  typedef typename ContainerT::size_type size_type;
  typedef typename ContainerT::value_type value_type;

  while ( true )
  {
    pos = str.find_first_of( delimiters, lastPos );
    if ( pos == std::string::npos )
    {
      pos = str.length();

      if ( ( pos != lastPos ) || ! trimEmpty )
      {
        tokens.push_back( value_type( str.data() + lastPos,
                                      (size_type)pos - lastPos ) );
      }

      break;
    }
    else
    {
      if ( ( pos != lastPos ) || ! trimEmpty )
      {
        tokens.push_back( value_type( str.data() + lastPos,
                                      (size_type)pos - lastPos ) );
      }
    }

    lastPos = pos + 1;
  }
}

} } // end namespace

#endif /* VITAL_TOKENIZE_H */
