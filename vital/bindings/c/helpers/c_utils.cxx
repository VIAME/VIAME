// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Common C Interface Utility implementation
 */

#include "c_utils.h"

namespace kwiver {
namespace vital_c {

void make_string_list( std::vector<std::string> const &list,
                       unsigned int &length, char ** &strings )
{
  length = static_cast< unsigned int >(list.size());
  strings = (char**)malloc( sizeof(char*) * length );
  for( unsigned int i = 0; i < length; ++i )
  {
    // +1 for null terminal
    strings[i] = (char*)malloc( sizeof(char) * (list[i].length() + 1) );
    std::strcpy( strings[i], list[i].c_str() );
  }
}

} } // end vital_c namespace
