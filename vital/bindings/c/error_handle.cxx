// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface common error handle structure implementation
 */

#include "error_handle.h"

#include <cstdlib>

/// Return a new, empty error handle object.
vital_error_handle_t* vital_eh_new()
{
  vital_error_handle_t* eh = (vital_error_handle_t*)std::malloc(sizeof(vital_error_handle_t));
  // Memory allocation may have failed.
  if( eh )
  {
    eh->error_code = 0;
    eh->message = (char*)0;
  }
  return eh;
}

/// Destroy the given non-null error handle structure pointer
void vital_eh_destroy( vital_error_handle_t *eh )
{
  if( eh )
  {
    if( eh->message )
    {
      std::free( eh->message );
    }
    std::free( eh );
  }
}
