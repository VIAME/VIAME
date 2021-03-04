// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::descriptor_set interface functions
 */

#ifndef VITAL_C_DESCRIPTOR_SET_H_
#define VITAL_C_DESCRIPTOR_SET_H_

#include <cstddef>

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/descriptor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// Base opaque descriptor instance type
typedef struct vital_descriptor_set_s vital_descriptor_set_t;

/// Create a new descriptor set from the array of descriptors.
/**
 * \param eh Vital error handle instance.
 */
VITAL_C_EXPORT
vital_descriptor_set_t*
vital_descriptor_set_new( vital_descriptor_t const **d_array,
                          size_t d_array_length,
                          vital_error_handle_t *eh );

/// Destroy a descriptor set
/**
 * \param ds Handle of the descriptor set instance to destroy.
 * \param eh Vital error handle instance.
 */
VITAL_C_EXPORT
void
vital_descriptor_set_destroy( vital_descriptor_set_t const *ds,
                              vital_error_handle_t *eh );

/// Get the size of a descriptor set
/**
 * \param ds The handle of the descriptor set instance.
 * \param eh Vital error handle instance.
 */
VITAL_C_EXPORT
size_t
vital_descriptor_set_size( vital_descriptor_set_t const *ds,
                           vital_error_handle_t *eh );

/// Get the descritpors stored in this set.
/**
 * \param ds The handle descriptor set instance.
 * \param[out] d_array Output array of descriptor instance handles. This array
 *   was created via malloc and the caller is responsible for freeing the
 *   array.
 * \param[out] d_array_length Output array length.
 * \param eh Vital error handle instance.
 */
VITAL_C_EXPORT
void
vital_descriptor_set_get_descriptors( vital_descriptor_set_t const *ds,
                                      vital_descriptor_t ***out_d_array,
                                      size_t *out_d_array_length,
                                      vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_DESCRIPTOR_SET_H_
