// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/config/config.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <sstream>

/**
 * \file config.cxx
 *
 * \brief Python bindings for \link kwiver::vital::config \endlink.
 */

namespace kwiver {
namespace vital {
namespace python {

kwiver::vital::config_block_value_t
config_block_set_value_cast( py::object const& value )
{
  return value.cast<std::string>();
}

void
config_set_value( kwiver::vital::config_block_sptr self,
                  kwiver::vital::config_block_key_t const&  key,
                  kwiver::vital::config_block_key_t const&  value )
{
  self->set_value< kwiver::vital::config_block_value_t > ( key, value );
}

kwiver::vital::config_block_value_t
config_get_value( kwiver::vital::config_block_sptr self,
                  kwiver::vital::config_block_key_t const&  key )
{
  return self->get_value< kwiver::vital::config_block_value_t > ( key );
}

kwiver::vital::config_block_value_t
config_get_value_with_default( kwiver::vital::config_block_sptr self,
                               kwiver::vital::config_block_key_t const&   key,
                               kwiver::vital::config_block_value_t const& def )
{
  return self->get_value< kwiver::vital::config_block_value_t > ( key, def );
}

pybind11::size_t
config_len( kwiver::vital::config_block_sptr self )
{
  return self->available_values().size();
}

kwiver::vital::config_block_value_t
config_getitem( kwiver::vital::config_block_sptr self,
                kwiver::vital::config_block_key_t const&  key )
{
  kwiver::vital::config_block_value_t val;

  try
  {
    val = config_get_value( self, key );
  }
  catch ( kwiver::vital::no_such_configuration_value_exception const& )
  {
    std::ostringstream sstr;

    sstr << "\'" << key << "\'";

    PyErr_SetString( PyExc_KeyError, sstr.str().c_str() );
    throw py::error_already_set();
  }

  return val;
}

void
config_setitem( kwiver::vital::config_block_sptr self,
                kwiver::vital::config_block_key_t const& key,
                py::object const& value )
{
  kwiver::vital::config_block_key_t const& str_value = py::str(value);

  self->set_value( key, str_value );
}

void
config_delitem( kwiver::vital::config_block_sptr self,
                kwiver::vital::config_block_key_t const& key )
{
  try
  {
    self->unset_value( key );
  }
  catch ( kwiver::vital::no_such_configuration_value_exception const& )
  {
    std::ostringstream sstr;

    sstr << "\'" << key << "\'";

    PyErr_SetString( PyExc_KeyError, sstr.str().c_str() );
    throw py::error_already_set();
  }
}

void
config( py::module& m)
{
  m.doc() = R"pbdoc(
                     Config module for vital
                     -----------------------

                     .. currentmodule:: config

                     .. autosummary::
                        :toctree: _generate

                     empty_config
                     ConfigKeys
                     Config
                   )pbdoc";
  m.def("empty_config", &kwiver::vital::config_block::empty_config
    , py::arg("name") = kwiver::vital::config_block_key_t()
    , R"pbdoc(Returns an empty :class:`kwiver.vital.config.Config` object)pbdoc");

  py::bind_vector<std::vector<std::string> >(m, "ConfigKeys"
    , R"pbdoc(A collection of keys for a configuration.)pbdoc");

  py::class_<kwiver::vital::config_block, kwiver::vital::config_block_sptr>(m, "Config"
    , R"pbdoc("A key-value store of configuration values)pbdoc")

    .def("subblock", &kwiver::vital::config_block::subblock
      , py::arg("name")
      , R"pbdoc(Returns a :class:`kwiver.vital.config.Config`
                from the configuration using the name of the subblock. The object
                is a copy of the block in the configuration.
                :param name: The name of the subblock in a :class:`kwiver.vital.config.Config` object
                :return: a subblock of type :class:`kwiver.vital.config.Config`
               )pbdoc")
    .def("subblock_view", &kwiver::vital::config_block::subblock_view
      , py::arg("name")
      , R"pbdoc(Returns a :class:`kwiver.vital.config.Config`
                from the configuration using the name of the subblock. The object
                is a view rather than the copy of the block in the configuration.
                :param name: The name of the subblock in a :class:`kwiver.vital.config.Config` object
                :return: a subblock of type :class:`kwiver.vital.config.Config`
               )pbdoc")
    .def("get_value", &kwiver::vital::python::config_get_value
      , py::arg("key")
      , R"pbdoc(Retrieve a value from the configuration using key.
                :param key: key in the configuration
                :return: A string value associated with the key
               )pbdoc")
    .def("get_value", &kwiver::vital::python::config_get_value_with_default
      , py::arg("key"), py::arg("default")
      , R"pbdoc(Retrieve a value from the configuration, using a default in case of failure.
                :param key: A key in the configuration
                :param default: A default value for the key
                :return: A string value associated with the key
               )pbdoc")
    .def("set_value", &kwiver::vital::python::config_set_value
      , py::arg("key"), py::arg("value")
      , R"pbdoc(Set a value in the configuration.
                :param key: A key in the configuration.
                :param value: A value in the configuration.
                :return: None
               )pbdoc")
    .def("unset_value", &kwiver::vital::config_block::unset_value
      , py::arg("key")
      , R"pbdoc(Unset a value in the configuration.
                :param key: A key in the configuration
                :return: None
               )pbdoc")
    .def("is_read_only", &kwiver::vital::config_block::is_read_only
      , py::arg("key")
      , R"pbdoc(Check if a key is marked as read only.
                :param key: A key in the configuration
                :return: Boolean specifying if the key value pair is read only
               )pbdoc")
    .def("mark_read_only", &kwiver::vital::config_block::mark_read_only
      , py::arg("key")
      , R"pbdoc(Mark a key as read only.
                :param key: A key in the configuration.
                :return: None
               )pbdoc")
    .def("merge_config", &kwiver::vital::config_block::merge_config
      , py::arg("config")
      , R"pbdoc(Merge another configuration block into the current one.
                :param config: An object of :class:`vital.config.Config`
                :return: An object of :class:`vital.config.Config` containing the merged configuration
               )pbdoc")
    .def("available_values", &kwiver::vital::config_block::available_values
      , R"pbdoc(Retrieves the list of available values in the configuration.
                :return: A list of string with all the keys
               )pbdoc")
    .def("has_value", &kwiver::vital::config_block::has_value
      , py::arg("key")
      , R"pbdoc(Returns True if the key is set.
                :param key: A key in the configuration
                :return: Boolean specifying if the key is present in the configuration
               )pbdoc")
    .def_static("block_sep", &kwiver::vital::config_block::block_sep
      , "The string which separates block names from key names.")
    .def_static("global_value", &kwiver::vital::config_block::global_value
      , "A special key which is automatically inherited on subblock requests.")
    .def("__len__", &kwiver::vital::python::config_len,
        R"pbdoc(Magic function that return the length of the configuration block)pbdoc")
    .def("__contains__", &kwiver::vital::config_block::has_value,
        R"pbdoc(Magic function to check if an key is in the configuration)pbdoc")
    .def("__getitem__", &kwiver::vital::python::config_getitem,
        R"pbdoc(Magic function to get a value)pbdoc")
    .def("__setitem__", &kwiver::vital::python::config_setitem,
        R"pbdoc(Magic function to assign a new value to a key)pbdoc")
    .def("__delitem__", &kwiver::vital::python::config_delitem,
        R"pbdoc(Magic function to remove a key)pbdoc");

// ----------------------------------------------------------------------------"
    py::class_<kwiver::vital::config_difference, std::shared_ptr<kwiver::vital::config_difference>>(m, "ConfigDifference"
        , "Represents difference between two config blocks" )

    .def(py::init<kwiver::vital::config_block_sptr, kwiver::vital::config_block_sptr>()
         , py::doc("Determine difference between config blocks"))
    .def(py::init<kwiver::vital::config_block_keys_t, kwiver::vital::config_block_sptr>()
         , py::doc("Determine difference between config blocks"))

    .def("extra_keys", &kwiver::vital::config_difference::extra_keys
         , "Return list of config keys that are not in the ref config")

    .def("unspecified_keys", &kwiver::vital::config_difference::unspecified_keys
         , "Return list of config keys that are in reference config but not in the other config")
;

}
}
}
}
