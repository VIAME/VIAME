/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/config.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
#include <boost/python/module.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/str.hpp>

#include <sstream>

/**
 * \file config.cxx
 *
 * \brief Python bindings for \link vistk::config\endlink.
 */

using namespace boost::python;

static vistk::config::value_t config_get_value(vistk::config_t self, vistk::config::key_t const& key);
static vistk::config::value_t config_get_value_with_default(vistk::config_t self, vistk::config::key_t const& key, vistk::config::value_t const& def);
static size_t config_len(vistk::config_t self);
static vistk::config::value_t config_getitem(vistk::config_t self, vistk::config::key_t const& key);
static void config_setitem(vistk::config_t self, vistk::config::key_t const& key, object const& value);
static void config_delitem(vistk::config_t self, vistk::config::key_t const& key);

static void translator(vistk::configuration_exception const& e);

BOOST_PYTHON_MODULE(config)
{
  register_exception_translator<
    vistk::configuration_exception>(translator);

  def("empty_config", &vistk::config::empty_config
    , (arg("name") = vistk::config::key_t())
    , "Returns an empty configuration.");

  class_<vistk::config::key_t>("ConfigKey"
    , "A key for a configuration.");
  class_<vistk::config::keys_t>("ConfigKeys"
    , "A collection of keys for a configuration.")
    .def(vector_indexing_suite<vistk::config::keys_t>())
  ;
  class_<vistk::config::value_t>("ConfigValue"
    , "A value in the configuration.");

  class_<vistk::config, vistk::config_t, boost::noncopyable>("Config"
    , "A key-value store of configuration values"
    , no_init)
    .def("subblock", &vistk::config::subblock
      , (arg("name"))
      , "Returns a subblock from the configuration.")
    .def("subblock_view", &vistk::config::subblock_view
      , (arg("name"))
      , "Returns a linked subblock from the configuration.")
    .def("get_value", &config_get_value
      , (arg("key"))
      , "Retrieve a value from the configuration.")
    .def("get_value", &config_get_value_with_default
      , (arg("key"), arg("default"))
      , "Retrieve a value from the configuration, using a default in case of failure.")
    .def("set_value", &vistk::config::set_value
      , (arg("key"), arg("value"))
      , "Set a value in the configuration.")
    .def("unset_value", &vistk::config::unset_value
      , (arg("key"))
      , "Unset a value in the configuration.")
    .def("is_read_only", &vistk::config::is_read_only
      , (arg("key"))
      , "Check if a key is marked as read only.")
    .def("mark_read_only", &vistk::config::mark_read_only
      , (arg("key"))
      , "Mark a key as read only.")
    .def("merge_config", &vistk::config::merge_config
      , (arg("config"))
      , "Merge another configuration block into the current one.")
    .def("available_values", &vistk::config::available_values
      , "Retrieves the list of available values in the configuration.")
    .def("has_value", &vistk::config::has_value
      , (arg("key"))
      , "Returns True if the key is set.")
    .def_readonly("block_sep", &vistk::config::block_sep
      , "The string which separates block names from key names.")
    .def_readonly("global_value", &vistk::config::global_value
      , "A special key which is automatically inherited on subblock requests.")
    .def("__len__", &config_len)
    .def("__contains__", &vistk::config::has_value)
    .def("__getitem__", &config_getitem)
    .def("__setitem__", &config_setitem)
    .def("__delitem__", &config_delitem)
  ;
}

vistk::config::value_t
config_get_value(vistk::config_t self, vistk::config::key_t const& key)
{
  return self->get_value<vistk::config::value_t>(key);
}

vistk::config::value_t
config_get_value_with_default(vistk::config_t self, vistk::config::key_t const& key, vistk::config::value_t const& def)
{
  return self->get_value<vistk::config::value_t>(key, def);
}

size_t
config_len(vistk::config_t self)
{
  return self->available_values().size();
}

vistk::config::value_t
config_getitem(vistk::config_t self, vistk::config::key_t const& key)
{
  vistk::config::value_t val;

  try
  {
    val = config_get_value(self, key);
  }
  catch (vistk::no_such_configuration_value_exception&)
  {
    std::ostringstream sstr;

    sstr << "\'" << key << "\'";

    PyErr_SetString(PyExc_KeyError, sstr.str().c_str());
    throw_error_already_set();
  }

  return val;
}

void
config_setitem(vistk::config_t self, vistk::config::key_t const& key, object const& value)
{
  self->set_value(key, extract<vistk::config::value_t>(str(value)));
}

void
config_delitem(vistk::config_t self, vistk::config::key_t const& key)
{
  try
  {
    self->unset_value(key);
  }
  catch (vistk::no_such_configuration_value_exception&)
  {
    std::ostringstream sstr;

    sstr << "\'" << key << "\'";

    PyErr_SetString(PyExc_KeyError, sstr.str().c_str());
    throw_error_already_set();
  }
}

void
translator(vistk::configuration_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
