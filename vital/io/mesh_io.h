/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief File IO functions for a \ref kwiver::vital::mesh
 *
 * Functions provide IO in multiple formats including OBJ, PLY, KML
 */

#ifndef VITAL_MESH_IO_H_
#define VITAL_MESH_IO_H_

#include <vital/vital_config.h>
#include <vital/vital_export.h>

#include <vital/types/mesh.h>

namespace kwiver {
namespace vital {


/// Read a mesh from a file, determine type from extension
VITAL_EXPORT
mesh_sptr read_mesh(const std::string& filename);


/// Read a mesh from a PLY file
VITAL_EXPORT
mesh_sptr read_ply(const std::string& filename);

/// Read a mesh from a PLY file
VITAL_EXPORT
mesh_sptr read_ply(std::istream& is);


/// Read a mesh from a PLY2 stream
VITAL_EXPORT
mesh_sptr read_ply2(std::istream& is);

/// Read a mesh from a PLY2 file
VITAL_EXPORT
mesh_sptr read_ply2(const std::string& filename);

/// Write a mesh to a PLY2 stream
VITAL_EXPORT
void write_ply2(std::ostream& os, const mesh& mesh);

/// Write a mesh to a PLY2 file
VITAL_EXPORT
void write_ply2(const std::string& filename, const mesh& mesh);


/// Read texture coordinates from a UV2 stream
VITAL_EXPORT
bool read_uv2(std::istream& is, mesh& mesh);

/// Read texture coordinates from a UV2 file
VITAL_EXPORT
bool read_uv2(const std::string& filename, mesh& mesh);


/// Read a mesh from a wavefront OBJ stream
VITAL_EXPORT
mesh_sptr read_obj(std::istream& is);

/// Read a mesh from a wavefront OBJ file
VITAL_EXPORT
mesh_sptr read_obj(const std::string& filename);

/// Write a mesh to a wavefront OBJ stream
VITAL_EXPORT
void write_obj(std::ostream& os, const mesh& mesh);

/// Write a mesh to a wavefront OBJ file
VITAL_EXPORT
void write_obj(const std::string& filename, const mesh& mesh);


/// Write a mesh into a kml stream
VITAL_EXPORT
void write_kml(std::ostream& os, const mesh& mesh);

/// Write a mesh into a kml file
VITAL_EXPORT
void write_kml(const std::string& filename, const mesh& mesh);


/// Write a mesh into a kml collada stream
VITAL_EXPORT
void write_kml_collada(std::ostream& os, const mesh& mesh);

/// Write a mesh into a kml collada file
VITAL_EXPORT
void write_kml_collada(const std::string& filename, const mesh& mesh);


/// Write a mesh into a vrml stream
VITAL_EXPORT
void write_vrml(std::ostream& os, const mesh& mesh);

/// Write a mesh into a vrml file
VITAL_EXPORT
void write_vrml(const std::string& filename, const mesh& mesh);


} // end namespace vital
} // end namespace kwiver

#endif // VITAL_MESH_IO_H_
