// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/processes/adapters/adapter_data_set.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>

// Type conversions
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/track_set.h>
#include <vital/types/object_track_set.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/timestamp.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/homography_f2f.h>

#include <memory>

PYBIND11_MAKE_OPAQUE(std::vector<unsigned char>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);

namespace ka = ::kwiver::adapter;
namespace py = pybind11;
namespace kwiver{
namespace sprokit{
namespace python{

// Accept a generic python object and cast to correct type before adding.
// This keeps a C++ process from having to deal with a py::object.
// We'll also accept datums directly so users can always use the index operator.
// When these are updated to add more types, the same will need to be done for datum
void add_value_correct_type(ka::adapter_data_set &self, ::sprokit::process::port_t const& port, py::object obj)
{

  if (obj.is_none())
  {
    throw py::type_error("Cannot add NoneType to adapter_data_set");
  }

  if (py::isinstance<::sprokit::datum>(obj))
  {
    ::sprokit::datum_t casted_obj = obj.cast<::sprokit::datum_t>();
    self.add_datum(port, casted_obj);
    return;
  }

  #define ADS_ADD_OBJECT(PYTYPE, TYPE) \
  if (py::isinstance<PYTYPE>(obj)) \
  { \
    TYPE casted_obj = obj.cast<TYPE>(); \
    self.add_value<TYPE>(port, casted_obj); \
    return; \
  }

  ADS_ADD_OBJECT(py::int_, int)
  ADS_ADD_OBJECT(py::float_, float)
  ADS_ADD_OBJECT(py::str, std::string)
  ADS_ADD_OBJECT(kwiver::vital::image_container, std::shared_ptr<kwiver::vital::image_container>)
  ADS_ADD_OBJECT(kwiver::vital::descriptor_set, std::shared_ptr<kwiver::vital::descriptor_set>)
  ADS_ADD_OBJECT(kwiver::vital::detected_object_set, std::shared_ptr<kwiver::vital::detected_object_set>)
  ADS_ADD_OBJECT(kwiver::vital::track_set, std::shared_ptr<kwiver::vital::track_set>)
  ADS_ADD_OBJECT(kwiver::vital::feature_track_set, std::shared_ptr<kwiver::vital::feature_track_set>)
  ADS_ADD_OBJECT(kwiver::vital::object_track_set, std::shared_ptr<kwiver::vital::object_track_set>)
  ADS_ADD_OBJECT(std::vector<double>, std::shared_ptr<std::vector<double>>)
  ADS_ADD_OBJECT(std::vector<std::string>, std::shared_ptr<std::vector<std::string>>)
  ADS_ADD_OBJECT(std::vector<unsigned char>, std::shared_ptr<std::vector<unsigned char>>)
  ADS_ADD_OBJECT(kwiver::vital::bounding_box_d, kwiver::vital::bounding_box_d)
  ADS_ADD_OBJECT(kwiver::vital::timestamp, kwiver::vital::timestamp)
  ADS_ADD_OBJECT(kwiver::vital::geo_polygon, kwiver::vital::geo_polygon)
  ADS_ADD_OBJECT(kwiver::vital::f2f_homography, kwiver::vital::f2f_homography)

  #undef ADS_ADD_OBJECT

  throw py::type_error("Unable to add object to adapter data set");
}

// Take data of an unknown type from a port and return. Can't return as an "any" object,
// so need to cast.
//
// The 'any.is_type<TYPE>()' call essentially does a string comparison
// of the underlying C++ types. This allows for comparisons across pybind modules.
// For example, adding a datum containing a datum.VectorDouble to an ADS then immediately
// retrieving the value at that port will return an adapter_data_set.VectorDouble. The Python
// types are different, but the C++ types are the same. Note that adding a datum.VectorDouble
// directly will not work, due to how Pybind/Python handles opaque types. Add an
// adapter_data_set.VectorDouble instead, or add a datum containing a datum.VectorDouble.
// Comparing type_info hashes will not work if the types are defined in different modules.
// See this issue for more information: https://github.com/pybind/pybind11/issues/912
py::object get_port_data_correct_type(ka::adapter_data_set &self, ::sprokit::process::port_t const& port)
{
  kwiver::vital::any const any = self.get_port_data<kwiver::vital::any>(port);

  #define ADS_GET_OBJECT(TYPE) \
  if (any.is_type<TYPE>()) \
  { \
    return py::cast(kwiver::vital::any_cast<TYPE>(any)); \
  }

  ADS_GET_OBJECT(int)
  ADS_GET_OBJECT(float)
  ADS_GET_OBJECT(std::string)
  ADS_GET_OBJECT(std::shared_ptr<kwiver::vital::image_container>)
  ADS_GET_OBJECT(std::shared_ptr<kwiver::vital::descriptor_set>)
  ADS_GET_OBJECT(std::shared_ptr<kwiver::vital::detected_object_set>)
  ADS_GET_OBJECT(std::shared_ptr<kwiver::vital::track_set>)
  ADS_GET_OBJECT(std::shared_ptr<kwiver::vital::feature_track_set>)
  ADS_GET_OBJECT(std::shared_ptr<kwiver::vital::object_track_set>)
  ADS_GET_OBJECT(std::shared_ptr<std::vector<double>>)
  ADS_GET_OBJECT(std::shared_ptr<std::vector<std::string>>)
  ADS_GET_OBJECT(std::shared_ptr<std::vector<unsigned char>>)
  ADS_GET_OBJECT(kwiver::vital::bounding_box_d)
  ADS_GET_OBJECT(kwiver::vital::timestamp)
  ADS_GET_OBJECT(kwiver::vital::geo_polygon)
  ADS_GET_OBJECT(kwiver::vital::f2f_homography)

  #undef ADS_GET_OBJECT

  std::string msg("Unable to convert object found at adapter data set port: ");
  msg += port;
  msg += ". Data is of type: ";
  msg += any.type_name();
  throw py::type_error(msg);
}

}
}
}

PYBIND11_MODULE(adapter_data_set, m)
{
  // Here we're essentially creating bindings for vectors of different types.
  // Using pybind's automatic conversion of py::list <-> vector would cause issues.
  // For example, how should the list [5, 2] be interpreted? As a vector of doubles,
  // or as a vector of unsigned chars? To avoid this ambiguity, we're going to bind
  // instances of vectors with certain template types explicitly.
  // If the above list is to be interpreted as a vector of doubles,
  // we would now use adapter_data_set.VectorDouble([5, 2]) on the Python side.
  py::bind_vector<std::vector<unsigned char>, std::shared_ptr<std::vector<unsigned char>>>(m, "VectorUChar", py::module_local(true));
  py::bind_vector<std::vector<double>, std::shared_ptr<std::vector<double>>>(m, "VectorDouble", py::module_local(true));
  py::bind_vector<std::vector<std::string>, std::shared_ptr<std::vector<std::string>>>(m, "VectorString", py::module_local(true));

  py::enum_<ka::adapter_data_set::data_set_type>(m, "DataSetType"
    , "Type of data set.")
    .value("data", ka::adapter_data_set::data)
    .value("end_of_input", ka::adapter_data_set::end_of_input)
  ;

  py::class_< ka::adapter_data_set, std::shared_ptr<ka::adapter_data_set > > ads(m, "AdapterDataSet");
    ads.def_static("create", &ka::adapter_data_set::create
        , (py::arg("type") = ka::adapter_data_set::data_set_type::data))

    .def("__iter__", [](ka::adapter_data_set &self)
    {
      return py::make_iterator(self.cbegin(), self.cend());
    }, py::keep_alive<0,1>())

    // Members
    .def("type", &ka::adapter_data_set::type)
    .def("is_end_of_data", &ka::adapter_data_set::is_end_of_data)

    // General add_value which adds any type, and __setitem__
    .def("add_value", &kwiver::sprokit::python::add_value_correct_type,
          "This method is equivalent to using __setitem__")
    .def("__setitem__", &kwiver::sprokit::python::add_value_correct_type)

    .def("add_datum", &ka::adapter_data_set::add_datum)

    // General get_value which gets data of any type from a port and __getitem__
    .def("get_port_data", &kwiver::sprokit::python::get_port_data_correct_type,
          "This method is equivalent to using __getitem__")
    .def("__getitem__", &kwiver::sprokit::python::get_port_data_correct_type)

    // The add_value function is templated.
    // To bind the function, we must bind explicit instances of it,
    // each with a different type.
    // First the native C++ types
    .def("_add_int", &ka::adapter_data_set::add_value<int>)
    .def("_add_float", &ka::adapter_data_set::add_value<float>)
    .def("_add_string", &ka::adapter_data_set::add_value<std::string>)
    // Next shared ptrs to kwiver vital types
    .def("_add_image_container", &ka::adapter_data_set::add_value<std::shared_ptr<kwiver::vital::image_container > >,
      py::arg("port"), py::arg("val").none(false))
    .def("_add_descriptor_set", &ka::adapter_data_set::add_value<std::shared_ptr<kwiver::vital::descriptor_set > >,
      py::arg("port"), py::arg("val").none(false))
    .def("_add_detected_object_set", &ka::adapter_data_set::add_value<std::shared_ptr<kwiver::vital::detected_object_set > >,
      py::arg("port"), py::arg("val").none(false))
    .def("_add_track_set", &ka::adapter_data_set::add_value<std::shared_ptr<kwiver::vital::track_set > >,
      py::arg("port"), py::arg("val").none(false))
    .def("_add_feature_track_set", &ka::adapter_data_set::add_value<std::shared_ptr<kwiver::vital::feature_track_set > >,
      py::arg("port"), py::arg("val").none(false))
    .def("_add_object_track_set", &ka::adapter_data_set::add_value<std::shared_ptr<kwiver::vital::object_track_set > >,
      py::arg("port"), py::arg("val").none(false))
    // Next shared ptrs to native C++ types
    .def("_add_double_vector", &ka::adapter_data_set::add_value<std::shared_ptr<std::vector<double>>>,
      py::arg("port"), py::arg("val").none(false))
    .def("_add_string_vector", &ka::adapter_data_set::add_value<std::shared_ptr<std::vector<std::string>>>,
      py::arg("port"), py::arg("val").none(false))
    .def("_add_uchar_vector", &ka::adapter_data_set::add_value<std::shared_ptr<std::vector<unsigned char>>>,
      py::arg("port"), py::arg("val").none(false))
    // Next kwiver vital types
    .def("_add_bounding_box", &ka::adapter_data_set::add_value<kwiver::vital::bounding_box_d>)
    .def("_add_timestamp", &ka::adapter_data_set::add_value<kwiver::vital::timestamp>)
    .def("_add_corner_points", &ka::adapter_data_set::add_value<kwiver::vital::geo_polygon>)
    .def("_add_f2f_homography", &ka::adapter_data_set::add_value<kwiver::vital::f2f_homography>)

    .def("empty", &ka::adapter_data_set::empty)

    // get_port_data is also templated
    .def("_get_port_data_int", &ka::adapter_data_set::get_port_data<int>)
    .def("_get_port_data_float", &ka::adapter_data_set::get_port_data<float>)
    .def("_get_port_data_string", &ka::adapter_data_set::get_port_data<std::string>)
    // Next shared ptrs to kwiver vital types
    .def("_get_port_data_image_container", &ka::adapter_data_set::get_port_data<std::shared_ptr<kwiver::vital::image_container > >)
    .def("_get_port_data_descriptor_set", &ka::adapter_data_set::get_port_data<std::shared_ptr<kwiver::vital::descriptor_set > >)
    .def("_get_port_data_detected_object_set", &ka::adapter_data_set::get_port_data<std::shared_ptr<kwiver::vital::detected_object_set > >)
    .def("_get_port_data_track_set", &ka::adapter_data_set::get_port_data<std::shared_ptr<kwiver::vital::track_set > >)
    .def("_get_port_data_feature_track_set", &ka::adapter_data_set::get_port_data<std::shared_ptr<kwiver::vital::feature_track_set > >)
    .def("_get_port_data_object_track_set", &ka::adapter_data_set::get_port_data<std::shared_ptr<kwiver::vital::object_track_set > >)
    //Next shared ptrs to native C++ types
    .def("_get_port_data_double_vector", &ka::adapter_data_set::get_port_data<std::shared_ptr<std::vector<double>>>)
    .def("_get_port_data_string_vector", &ka::adapter_data_set::get_port_data<std::shared_ptr<std::vector<std::string>>>)
    .def("_get_port_data_uchar_vector", &ka::adapter_data_set::get_port_data<std::shared_ptr<std::vector<unsigned char>>>)
    //Next kwiver vital types
    .def("_get_port_data_bounding_box", &ka::adapter_data_set::get_port_data<kwiver::vital::bounding_box_d>)
    .def("_get_port_data_timestamp", &ka::adapter_data_set::get_port_data<kwiver::vital::timestamp>)
    .def("_get_port_data_corner_points", &ka::adapter_data_set::get_port_data<kwiver::vital::geo_polygon>)
    .def("_get_port_data_f2f_homography", &ka::adapter_data_set::get_port_data<kwiver::vital::f2f_homography>)
    .def("__nice__", [](const ka::adapter_data_set& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
      py::exec(R"(
          retval = 'size={}'.format(len(self))
      )", py::globals(), locals);
      return locals["retval"].cast<std::string>();
      })
  .def("__repr__", [](py::object& self) -> std::string {
      auto locals = py::dict(py::arg("self")=self);
      py::exec(R"(
          classname = self.__class__.__name__
          devnice = self.__nice__()
          retval = '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
      )", py::globals(), locals);
      return locals["retval"].cast<std::string>();
      })
  .def("__str__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        from kwiver.sprokit.pipeline import datum

        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s)>\n' % (classname, devnice)
        retval += '\t{'
        for i, (port, datum_obj) in enumerate(self):
            if i:
                retval += ', '
            retval += port
            retval += ": "
            retval += str(datum_obj.get_datum())
        retval += '}'
    )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
    })
  .def("__len__", &ka::adapter_data_set::size)
  ;

  ads.doc() = R"(
      Python bindings for kwiver::adapter::adapter_data_set

      Example:
          >>> from kwiver.sprokit.adapters import adapter_data_set
          >>> # Following ads has type "data". We can add/get data to/from ports
          >>> ads = adapter_data_set.AdapterDataSet.create()
          >>> assert ads.type() == adapter_data_set.DataSetType.data
          >>> # Can add as a general python object
          >>> ads["port1"] = "a_string"
          >>> # Can also add by specifying type
          >>> ads._add_int("port2", 5)
          >>> # Get both values
          >>> assert ads["port1"] == "a_string"
          >>> assert ads._get_port_data_int("port2") == 5
      )";
}
