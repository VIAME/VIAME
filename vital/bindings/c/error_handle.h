// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface common error handle structure
 */

#ifndef VITAL_C_ERROR_HANDLE_H_
#define VITAL_C_ERROR_HANDLE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/vital_c_export.h>

/// Common error handle structure
/**
 * When an instance of this structure is passed into a Vital API and an error
 * occurs a new string (char*) is allocated for ``message`` and the error_code
 * is set to a non-zero value.
 *
 * If ``message`` is already allocated then the previous memory is first freed.
 *
 * A single error handle can be reused between multiple API calls, but one
 * should check the status between calls and copy the message string before
 * reusing the error handle to avoid losing the message.
 */
typedef struct vital_error_handle_s {
  int error_code;
  char *message;
} vital_error_handle_t;

/// Return a new, empty error handle object.
/**
 * \returns New error handle whose error code is set to 0 and message to NULL.
 */
VITAL_C_EXPORT
vital_error_handle_t* vital_eh_new();

/// Destroy the given error handle structure pointer
/**
 * This function does nothing if passed a NULL pointer.
 *
 * \param eh Pointer to the error handle instance to destroy.
 */
VITAL_C_EXPORT
void vital_eh_destroy( vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_ERROR_HANDLE_H_
