/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \brief Function to generate \ref kwiver::vital::camera_rpc from metadata
 */

#ifndef VITAL_CAMERA_FROM_METADATA_H_
#define VITAL_CAMERA_FROM_METADATA_H_

#include <vital/vital_export.h>

#include <vital/types/camera_rpc.h>
#include <vital/types/metadata.h>

namespace kwiver {
namespace vital {

/// Convert space separated sting to Eigen vector
/**
 * \param s The string to be converted.
 * \return The converted vector.
 */
Eigen::VectorXd
VITAL_EXPORT string_to_vector( std::string const& s );

/// Produce RPC camera from metadata
/**
 * \param file_path   The path to the file to read in.
 * \return A new camera object representing the contents of the read-in file.
 */
camera_sptr
VITAL_EXPORT camera_from_metadata( metadata_sptr const& md );


} } // end namespace

#endif // VITAL_CAMERA_FROM_METADATA_H_
