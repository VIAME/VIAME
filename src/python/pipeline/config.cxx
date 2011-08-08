/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/config.h>

#include <boost/python.hpp>

/**
 * \file config.cxx
 *
 * \brief Python bindings for \link vistk::config\endlink.
 */

using namespace boost::python;

static vistk::config_t empty_config();
static vistk::config_t named_empty_config(vistk::config::key_t const& name);
static vistk::config::value_t get_value(vistk::config_t self, vistk::config::key_t const& key);
static vistk::config::value_t get_value_with_default(vistk::config_t self, vistk::config::key_t const& key, vistk::config::value_t const& def);
static void translator(vistk::configuration_exception const& e);

BOOST_PYTHON_FUNCTION_OVERLOADS(empty_config_overloads, vistk::config::empty_config, 0, 1);

BOOST_PYTHON_MODULE(config)
{
  register_exception_translator<
    vistk::configuration_exception>(translator);

  def("empty_config", &vistk::config::empty_config
    , empty_config_overloads());

  class_<vistk::config, vistk::config_t, boost::noncopyable>("Config", no_init)
    .def("subblock", &vistk::config::subblock)
    .def("subblock_view", &vistk::config::subblock_view)
    .def("get_value", &get_value)
    .def("get_value", &get_value_with_default)
    .def("set_value", &vistk::config::set_value)
    .def("unset_value", &vistk::config::unset_value)
    .def("is_read_only", &vistk::config::is_read_only)
    .def("mark_read_only", &vistk::config::mark_read_only)
    .def("merge_config", &vistk::config::merge_config)
    .def("available_values", &vistk::config::available_values)
    .def("has_value", &vistk::config::has_value)
    .def_readonly("block_sep", &vistk::config::block_sep)
    .def_readonly("global_value", &vistk::config::global_value)
  ;
}

vistk::config_t
empty_config()
{
  return vistk::config::empty_config();
}

vistk::config_t
named_empty_config(vistk::config::key_t const& name)
{
  return vistk::config::empty_config(name);
}

vistk::config::value_t
get_value(vistk::config_t self, vistk::config::key_t const& key)
{
  return self->get_value<vistk::config::value_t>(key);
}

vistk::config::value_t
get_value_with_default(vistk::config_t self, vistk::config::key_t const& key, vistk::config::value_t const& def)
{
  return self->get_value<vistk::config::value_t>(key, def);
}

void
translator(vistk::configuration_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
