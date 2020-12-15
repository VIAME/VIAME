// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for detected_object_type class
 */

#ifndef VITAL_C_DETECTED_OBJECT_TYPE_H_
#define VITAL_C_DETECTED_OBJECT_TYPE_H_

#ifdef __cplusplus
extern "C"
{
#endif
#include <vital/bindings/c/vital_c_export.h>
#include <stddef.h>

/// VITAL detected object type opaque structure
typedef struct vital_detected_object_type_s vital_detected_object_type_t;

VITAL_C_EXPORT
vital_detected_object_type_t* vital_detected_object_type_new();

VITAL_C_EXPORT
void vital_detected_object_type_destroy(vital_detected_object_type_t* obj);

VITAL_C_EXPORT
vital_detected_object_type_t* vital_detected_object_type_new_from_list( vital_detected_object_type_t* obj,
                                                                        size_t count,
                                                                        char** class_names,
                                                                        double* scores);

VITAL_C_EXPORT
bool vital_detected_object_type_has_class_name( vital_detected_object_type_t* obj, char* class_name );

VITAL_C_EXPORT
double vital_detected_object_type_score( vital_detected_object_type_t* obj, char* class_name );

VITAL_C_EXPORT
char* vital_detected_object_type_get_most_likely_class( vital_detected_object_type_t* obj);

VITAL_C_EXPORT
double vital_detected_object_type_get_most_likely_score( vital_detected_object_type_t* obj);

VITAL_C_EXPORT
void vital_detected_object_type_set_score( vital_detected_object_type_t* obj, char* class_name, double score);

VITAL_C_EXPORT
void vital_detected_object_type_delete_score( vital_detected_object_type_t* obj, char* class_name);

VITAL_C_EXPORT
char** vital_detected_object_type_class_names( vital_detected_object_type_t* obj, double thresh );

VITAL_C_EXPORT
char** vital_detected_object_type_all_class_names(vital_detected_object_type_t* obj);

#ifdef __cplusplus
}
#endif

#endif
