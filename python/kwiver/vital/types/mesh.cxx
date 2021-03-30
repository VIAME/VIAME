// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/io/mesh_io.h>
#include <vital/types/mesh.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mesh, m)
{
  py::class_<kwiver::vital::mesh,
             std::shared_ptr<kwiver::vital::mesh> >(m, "Mesh")
    .def(py::init<>())
    .def("is_init",   &kwiver::vital::mesh::is_init)
    .def("num_verts", [](kwiver::vital::mesh& self)
    {
      if(self.is_init())
      {
        return self.num_verts();
      }
      return (unsigned int)0;
    })
    .def("num_faces", [](kwiver::vital::mesh& self)
    {
      if(self.is_init())
      {
        return self.num_faces();
      }
      return (unsigned int)0;
    })
    .def("num_edges", [](kwiver::vital::mesh& self)
    {
      if(self.is_init())
      {
        return self.num_edges();
      }
      return (unsigned int)0;
    })
    .def_static("from_ply_file", [](std::string path)
                                 {
                                   return kwiver::vital::read_ply(path);
                                 });
}
