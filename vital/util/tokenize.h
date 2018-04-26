/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
