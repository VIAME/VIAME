/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief C/C++ interface to vital::descriptor_set class
 */

#ifndef VITAL_C_DESCRIPTOR_SET_HXX_
#define VITAL_C_DESCRIPTOR_SET_HXX_

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/descriptor_set.h>
#include <vital/types/descriptor_set.h>


// -----------------------------------------------------------------------------
// These two functions are a bridge between C++ and the internal C smart pointer
// management.
// -----------------------------------------------------------------------------


/// Create a vital_descriptor_set_t around an existing shared pointer.
/**
 * If an error occurs, a NULL pointer is returned.
 *
 * \param ds Shared pointer to a vital::descriptor_set instance.
 * \param eh Vital error handle instance. May be null to ignore errors.
 */
VITAL_C_EXPORT
vital_descriptor_set_t*
vital_descriptor_set_new_from_sptr( kwiver::vital::descriptor_set_sptr ds_sptr,
                                    vital_error_handle_t* eh );


/// Get the vital::descriptor_set shared pointer for a handle.
/**
 * If an error occurs, an empty shared pointer is returned.
 *
 * \param ds Vital C handle to the descriptor_set instance to get the shared
 *   pointer reference of.
 * \param eh Vital error handle instance. May be null to ignore errors.
 */
VITAL_C_EXPORT
kwiver::vital::descriptor_set_sptr
vital_descriptor_set_to_sptr( vital_descriptor_set_t* ds,
                              vital_error_handle_t* eh );


#endif // VITAL_C_DESCRIPTOR_SET_HXX_
