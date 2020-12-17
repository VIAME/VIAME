// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Common C Interface Utilities
 */

#ifndef VITAL_C_COMMON_H_
#define VITAL_C_COMMON_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>

#include <vital/bindings/c/vital_c_export.h>

/// Simple string structure
typedef struct {
  size_t length;
  char *str;
} vital_string_t;

/// Allocate a new vital string structure
VITAL_C_EXPORT
vital_string_t* vital_string_new(size_t length, char const* s);

/// Free an allocated string structure
VITAL_C_EXPORT
void vital_string_free( vital_string_t *s );

/// Common function for freeing string lists
VITAL_C_EXPORT
void vital_common_free_string_list( size_t length, char **keys );

/// Other free functions
VITAL_C_EXPORT void vital_free_pointer( void *thing );
VITAL_C_EXPORT void vital_free_double_pointer( size_t length, void **things );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_COMMON_H_
