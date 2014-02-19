/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

#include <sprokit/pipeline/config.h>

#include <sprokit/python/util/python_gil.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/str.hpp>

#include <sstream>

/**
 * \file config.cxx
 *
 * \brief Python bindings for \link sprokit::config\endlink.
 */

using namespace boost::python;

static sprokit::config::value_t config_get_value(sprokit::config_t self, sprokit::config::key_t const& key);
static sprokit::config::value_t config_get_value_with_default(sprokit::config_t self, sprokit::config::key_t const& key, sprokit::config::value_t const& def);
static size_t config_len(sprokit::config_t self);
static sprokit::config::value_t config_getitem(sprokit::config_t self, sprokit::config::key_t const& key);
static void config_setitem(sprokit::config_t self, sprokit::config::key_t const& key, object const& value);
static void config_delitem(sprokit::config_t self, sprokit::config::key_t const& key);

BOOST_PYTHON_MODULE(config)
{
  def("empty_config", &sprokit::config::empty_config
    , (arg("name") = sprokit::config::key_t())
    , "Returns an empty configuration.");

  class_<sprokit::config::key_t>("ConfigKey"
    , "A key for a configuration.");
  class_<sprokit::config::keys_t>("ConfigKeys"
    , "A collection of keys for a configuration.")
    .def(vector_indexing_suite<sprokit::config::keys_t>())
  ;
  class_<sprokit::config::description_t>("ConfigDescription"
    , "A description of a configuration key.");
  class_<sprokit::config::value_t>("ConfigValue"
    , "A value in the configuration.");

  class_<sprokit::config, sprokit::config_t, boost::noncopyable>("Config"
    , "A key-value store of configuration values"
    , no_init)
    .def("subblock", &sprokit::config::subblock
      , (arg("name"))
      , "Returns a subblock from the configuration.")
    .def("subblock_view", &sprokit::config::subblock_view
      , (arg("name"))
      , "Returns a linked subblock from the configuration.")
    .def("get_value", &config_get_value
      , (arg("key"))
      , "Retrieve a value from the configuration.")
    .def("get_value", &config_get_value_with_default
      , (arg("key"), arg("default"))
      , "Retrieve a value from the configuration, using a default in case of failure.")
    .def("set_value", &sprokit::config::set_value
      , (arg("key"), arg("value"))
      , "Set a value in the configuration.")
    .def("unset_value", &sprokit::config::unset_value
      , (arg("key"))
      , "Unset a value in the configuration.")
    .def("is_read_only", &sprokit::config::is_read_only
      , (arg("key"))
      , "Check if a key is marked as read only.")
    .def("mark_read_only", &sprokit::config::mark_read_only
      , (arg("key"))
      , "Mark a key as read only.")
    .def("merge_config", &sprokit::config::merge_config
      , (arg("config"))
      , "Merge another configuration block into the current one.")
    .def("available_values", &sprokit::config::available_values
      , "Retrieves the list of available values in the configuration.")
    .def("has_value", &sprokit::config::has_value
      , (arg("key"))
      , "Returns True if the key is set.")
    .def_readonly("block_sep", &sprokit::config::block_sep
      , "The string which separates block names from key names.")
    .def_readonly("global_value", &sprokit::config::global_value
      , "A special key which is automatically inherited on subblock requests.")
    .def("__len__", &config_len)
    .def("__contains__", &sprokit::config::has_value)
    .def("__getitem__", &config_getitem)
    .def("__setitem__", &config_setitem)
    .def("__delitem__", &config_delitem)
  ;
}

sprokit::config::value_t
config_get_value(sprokit::config_t self, sprokit::config::key_t const& key)
{
  return self->get_value<sprokit::config::value_t>(key);
}

sprokit::config::value_t
config_get_value_with_default(sprokit::config_t self, sprokit::config::key_t const& key, sprokit::config::value_t const& def)
{
  return self->get_value<sprokit::config::value_t>(key, def);
}

size_t
config_len(sprokit::config_t self)
{
  return self->available_values().size();
}

sprokit::config::value_t
config_getitem(sprokit::config_t self, sprokit::config::key_t const& key)
{
  sprokit::config::value_t val;

  try
  {
    val = config_get_value(self, key);
  }
  catch (sprokit::no_such_configuration_value_exception const&)
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    std::ostringstream sstr;

    sstr << "\'" << key << "\'";

    PyErr_SetString(PyExc_KeyError, sstr.str().c_str());
    throw_error_already_set();
  }

  return val;
}

void
config_setitem(sprokit::config_t self, sprokit::config::key_t const& key, object const& value)
{
  sprokit::python::python_gil const gil;

  (void)gil;

  self->set_value(key, extract<sprokit::config::value_t>(str(value)));
}

void
config_delitem(sprokit::config_t self, sprokit::config::key_t const& key)
{
  try
  {
    self->unset_value(key);
  }
  catch (sprokit::no_such_configuration_value_exception const&)
  {
    sprokit::python::python_gil const gil;

    (void)gil;

    std::ostringstream sstr;

    sstr << "\'" << key << "\'";

    PyErr_SetString(PyExc_KeyError, sstr.str().c_str());
    throw_error_already_set();
  }
}
