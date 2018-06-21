/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief data_serializer algorithm definition instantiation
 */

#include <vital/algo/data_serializer.h>
#include <vital/algo/algorithm.txx>
#include <vital/util/string.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

const std::string data_serializer::DEFAULT_ELEMENT_NAME {"datum"};

// ----------------------------------------------------------------------------
data_serializer
::data_serializer()
{
  attach_logger( "data_serializer" );
}


// ----------------------------------------------------------------------------
bool
data_serializer
::check_element_names( serialize_param_t elements )
{
  for ( const auto it : elements )
  {
    if ( m_element_names.count( it.first ) == 0 )
    {
      // throw error
      std::stringstream str;
      str << "Element name \"" << it.first
          << "\" is not in the supported set. Supported elements are: "
          << kwiver::vital::join( m_element_names, ", " );

      throw std::runtime_error( str.str() );
    }
  } // end for

  // Check for all allowable names being in the map;
  return (elements.size() == m_element_names.size() );
}


// ----------------------------------------------------------------------------
const std::set< std::string >&
data_serializer
::element_names() const
{
  return m_element_names;
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::data_serializer);
/// \endcond
