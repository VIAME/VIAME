// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface to vital::detected_object_set class
 */

#ifndef VITAL_C_DETECTED_OBJECT_SET_H_
#define VITAL_C_DETECTED_OBJECT_SET_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/detected_object.h>
#include <vital/bindings/c/types/detected_object_set.h>

#include <stddef.h>

/// VITAL Image opaque structure
typedef struct vital_detected_object_set_s vital_detected_object_set_t;

VITAL_C_EXPORT
vital_detected_object_set_t* vital_detected_object_set_new();

VITAL_C_EXPORT
vital_detected_object_set_t* vital_detected_object_set_new_from_list( vital_detected_object_t** dobj,
                                                                      size_t n);

VITAL_C_EXPORT
void vital_detected_object_set_destroy( vital_detected_object_set_t* obj);

VITAL_C_EXPORT
void vital_detected_object_set_add( vital_detected_object_set_t* set,
                                    vital_detected_object_t* obj );

VITAL_C_EXPORT
size_t vital_detected_object_set_size( vital_detected_object_set_t* obj);

VITAL_C_EXPORT
void vital_detected_object_set_select_threshold( vital_detected_object_set_t* obj,
                                                 double thresh,
                                                 vital_detected_object_t*** output,
                                                 size_t* length );

VITAL_C_EXPORT
void vital_detected_object_set_select_class_threshold( vital_detected_object_set_t* obj,
                                                       const char* class_name,
                                                       double thresh,
                                                       vital_detected_object_t*** output,
                                                       size_t* length );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_DETECTED_OBJECT_SET_H_
