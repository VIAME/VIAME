// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief core mesh_container interface

#ifndef VITAL_MESH_CONTAINER_H_
#define VITAL_MESH_CONTAINER_H_

#include <viame/algorithm_framework/vital_config.h>

#include <viame/core_types/mesh.h>
#include <viame/core_types/metadata.h>

namespace kwiver {

namespace vital {

/// An abstract representation of a mesh container.
///
/// This class provides an interface for passing mesh data
/// between algorithms.  It is intended to be a wrapper for mesh
/// classes in third-party libraries and facilitate conversion between
/// various representations.  It provides limited access to the underlying
/// data and is not intended for direct use in mesh processing algorithms.
class mesh_container
{
public:
  virtual ~mesh_container() = default;

  /// Return the number of vertices in the mesh.
  virtual size_t num_verts() const = 0;

  /// Return the number of faces in the mesh.
  virtual size_t num_faces() const = 0;

  /// Return mesh/convert arrow's mesh representation to vital.
  virtual kwiver::vital::mesh mesh() const = 0;

  /// Return value indicates how texture coordinates are indexed.
  ///
  /// ON_VERT is one coordinate per vertex.
  /// ON_CORNER is one coordinate per half edge (i.e. corner).
  virtual kwiver::vital::mesh::tex_coord_type has_tex_coords() const = 0;

  /// Return the texture coordinates.
  ///
  /// Coordinates are ordered as indicated by has_tex_coords.
  virtual std::vector< vector_2d > tex_coords() const = 0;

  /// Set the texture coordinates.
  ///
  /// See vital::mesh::set_tex_coordinates for details.
  virtual void set_tex_coords( std::vector< vector_2d > const& tc ) = 0;
};

/// Shared pointer for base mesh_container type
using mesh_container_sptr = std::shared_ptr< mesh_container >;
using mesh_container_scptr = std::shared_ptr< mesh_container const >;

// ----------------------------------------------------------------------------
/// This concrete mesh container is simply a wrapper around a mesh.
class simple_mesh_container
  : public mesh_container
{
public:
  explicit simple_mesh_container( const kwiver::vital::mesh& d ) : data( d ) {}

  /// Return the number of vertices in the mesh.
  virtual size_t
  num_verts() const { return data.num_verts(); }

  /// Return the number of faces in the mesh.
  virtual size_t
  num_faces() const { return data.num_faces(); }

  /// Return an in-memory mesh class to access the data.
  virtual kwiver::vital::mesh
  mesh() const { return data; }

  /// Return a reference to the underlying mesh.
  kwiver::vital::mesh& mesh_ref() { return data; }

  /// Return a const reference to the underlying mesh.
  kwiver::vital::mesh const&
  mesh_ref() const { return data; }

  /// Return the texture coordinate status for the mesh.
  virtual kwiver::vital::mesh::tex_coord_type
  has_tex_coords() const
  {
    return data.has_tex_coords();
  }

  /// Return the texture coordinates.
  virtual std::vector< vector_2d >
  tex_coords() const
  {
    return data.tex_coords();
  }

  /// Set the texture coordinates.
  virtual void
  set_tex_coords( std::vector< vector_2d > const& tc )
  {
    data.set_tex_coords( tc );
  }

protected:
  kwiver::vital::mesh data;
};

} // namespace vital

} // namespace kwiver

#endif
