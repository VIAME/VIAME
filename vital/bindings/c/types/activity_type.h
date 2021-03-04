// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for activity_type class
 */

#ifndef VITAL_C_ACTIVITY_TYPE_H_
#define VITAL_C_ACTIVITY_TYPE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/vital_c_export.h>

#include <stddef.h>

/// VITAL Image opaque structure
typedef struct vital_activity_type_s vital_activity_type_t;

VITAL_C_EXPORT
vital_activity_type_t* vital_activity_type_new();

VITAL_C_EXPORT
void vital_activity_type_destroy(vital_activity_type_t* obj);

VITAL_C_EXPORT
vital_activity_type_t* vital_activity_type_new_from_list( vital_activity_type_t* obj,
                                                          size_t count,
                                                          char** class_names,
                                                          double* scores);

VITAL_C_EXPORT
bool vital_activity_type_has_class_name( vital_activity_type_t* obj, char* class_name );

VITAL_C_EXPORT
double vital_activity_type_score( vital_activity_type_t* obj, char* class_name );

VITAL_C_EXPORT
char* vital_activity_type_get_most_likely_class( vital_activity_type_t* obj);

VITAL_C_EXPORT
double vital_activity_type_get_most_likely_score( vital_activity_type_t* obj);

VITAL_C_EXPORT
void vital_activity_type_set_score( vital_activity_type_t* obj, char* class_name, double score);

VITAL_C_EXPORT
void vital_activity_type_delete_score( vital_activity_type_t* obj, char* class_name);

VITAL_C_EXPORT
char** vital_activity_type_class_names( vital_activity_type_t* obj, double thresh );

VITAL_C_EXPORT
char** vital_activity_type_all_class_names(vital_activity_type_t* obj);

#ifdef __cplusplus
}
#endif

#endif
