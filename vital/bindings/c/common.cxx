// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Common C Interface Utilities Implementation
 */

#include "common.h"

#include <cstdlib>
#include <cstring>

/// Allocate a new vital string structure
vital_string_t* vital_string_new(size_t length, char const* s)
{
  vital_string_t* n =
    (vital_string_t*)malloc(sizeof(vital_string_t));
  n->length = length;
  // When length 0, this is just a 1 character string that is just the null
  // byte.
  n->str = (char*)malloc(sizeof(char) * (length+1));
  n->str[length] = 0;

  if( length && s )
  {
    strncpy( n->str, s, length );
  }
  return n;
}

/// Free an alocated string structure
void vital_string_free( vital_string_t *s )
{
  free(s->str);
  free(s);
}

/// Common function for freeing string lists
void vital_common_free_string_list( size_t length,
                                    char **keys )
{
  for( unsigned int i = 0; i < length; i++ )
  {
    free(keys[i]);
  }
  free(keys);
}

void vital_free_pointer( void *thing )
{
  if( thing )
  {
    free(thing);
  }
}

void vital_free_double_pointer( size_t length, void **things )
{
  if( things )
  {
    for( size_t i=0; i < length; i++ )
    {
      if( things[i] )
      {
        free(things[i]);
      }
    }
    free(things);
  }
}
