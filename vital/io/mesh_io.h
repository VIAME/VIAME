// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
