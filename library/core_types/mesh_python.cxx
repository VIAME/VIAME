// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/io/mesh_io.h>
#include <viame/core_types/mesh.h>

#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace kv = kwiver::vital;
namespace py = pybind11;

PYBIND11_MODULE( mesh, m )
{
  py::class_< kv::mesh, std::shared_ptr< kv::mesh > >( m, "Mesh" )
    .def( py::init<>() )
    .def( "is_init",   &kv::mesh::is_init )
    .def(
      "num_verts", [](kv::mesh& self){
        if( self.is_init() )
        {
          return self.num_verts();
        }
        return ( unsigned int ) 0;
      } )
    .def(
      "num_faces", [](kv::mesh& self){
        if( self.is_init() )
        {
          return self.num_faces();
        }
        return ( unsigned int ) 0;
      } )
    .def(
      "num_edges", [](kv::mesh& self){
        if( self.is_init() )
        {
          return self.num_edges();
        }
        return ( unsigned int ) 0;
      } )
    .def( "has_tex_coords", &kv::mesh::has_tex_coords )
    .def( "tex_coords",   &kv::mesh::tex_coords )
    .def( "set_tex_source",   &kv::mesh::set_tex_source )
    .def( "texture_map", &kv::mesh::texture_map )
    .def( "compute_vertex_normals", &kv::mesh::compute_vertex_normals )
    .def(
      "compute_vertex_normals_from_faces",
      &kv::mesh::compute_vertex_normals_from_faces )
    .def(
      "compute_face_normals",
      &kv::mesh::compute_face_normals, py::arg( "norm" ) = true )
    .def(
      "face_normals", [](kv::mesh& self){
        return self.faces().normals();
      } )
    .def(
      "faces", [](kv::mesh& self){
        const kv::mesh_face_array_base& faces = self.faces();

        auto face_dim = faces.regularity();

        py::list retVal;
        for( unsigned int i = 0; i < faces.size(); ++i )
        {
          py::list verts;
          if( face_dim == 0 )
          {
            auto curr_face =
              static_cast< const kv::mesh_face_array& >( faces )[ i ];
            for( unsigned int j = 0; j < curr_face.size(); ++j )
            {
              verts.append( curr_face[ j ] );
            }
          }
          else if( face_dim == 3 )
          {
            auto curr_face =
              static_cast< const kv::mesh_regular_face_array< 3 >& >( faces )[ i ];
            for( unsigned int j = 0; j < curr_face.num_verts(); ++j )
            {
              verts.append( curr_face[ j ] );
            }
          }
          else if( face_dim == 4 )
          {
            auto curr_face =
              static_cast< const kv::mesh_regular_face_array< 4 >& >( faces )[ i ];
            for( unsigned int j = 0; j < curr_face.num_verts(); ++j )
            {
              verts.append( curr_face[ j ] );
            }
          }
          retVal.append( verts );
        }

        return retVal;
      } )
    .def(
      "vertices", [](kv::mesh& self){
        const kv::mesh_vertex_array_base& raw_verts = self.vertices();

        py::list retVal;
        for( unsigned int i = 0; i < raw_verts.size(); ++i )
        {
          py::list vert;
          for( unsigned int j = 0; j < raw_verts.dim(); ++j )
          {
            vert.append( raw_verts( i, j ) );
          }
          retVal.append( vert );
        }
        return retVal;
      } )
    .def_static(
      "from_ply_file", [](std::string path){
        return kv::read_ply( path );
      } )
    .def_static(
      "from_ply2_file", [](std::string path){
        return kv::read_ply2( path );
      } )
    .def_static(
      "from_obj_file", [](std::string path){
        return kv::read_obj( path );
      } )
    .def_static(
      "from_file", [](std::string path){
        return kv::read_mesh( path );
      } )
    .def_static(
      "to_ply2_file", [](std::string path, kv::mesh& mesh){
        return kv::write_ply2( path, mesh );
      } )
    .def_static(
      "to_obj_file", [](std::string path, kv::mesh& mesh){
        return kv::write_obj( path, mesh );
      } )
    .def_static(
      "to_kml_file", [](std::string path, kv::mesh& mesh){
        return kv::write_kml( path, mesh );
      } )
    .def_static(
      "to_kml_collada_file", [](std::string path, kv::mesh& mesh){
        return kv::write_kml_collada( path, mesh );
      } )
    .def_static(
      "to_vrml_file", [](std::string path, kv::mesh& mesh){
        return kv::write_vrml( path, mesh );
      } );

  py::enum_< kv::mesh::tex_coord_type >( m, "tex_coord_type" )
    .value( "TEX_COORD_NONE", kv::mesh::tex_coord_type::TEX_COORD_NONE )
    .value( "TEX_COORD_ON_VERT", kv::mesh::tex_coord_type::TEX_COORD_ON_VERT )
    .value(
      "TEX_COORD_ON_CORNER",
      kv::mesh::tex_coord_type::TEX_COORD_ON_CORNER );
}
