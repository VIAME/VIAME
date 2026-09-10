// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/config/module_config_helpers.h>
#include <viame/core_types/geo_polygon.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <sstream>

namespace kv = kwiver::vital;

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
  return value.cast< std::string >();
}

void
config_set_value(
  kwiver::vital::config_block_sptr self,
  kwiver::vital::config_block_key_t const&  key,
  kwiver::vital::config_block_key_t const&  value )
{
  self->set_value< kwiver::vital::config_block_value_t >( key, value );
}

kwiver::vital::config_block_value_t
config_get_value(
  kwiver::vital::config_block_sptr self,
  kwiver::vital::config_block_key_t const&  key )
{
  return self->get_value< kwiver::vital::config_block_value_t >( key );
}

kwiver::vital::config_block_value_t
config_get_value_with_default(
  kwiver::vital::config_block_sptr self,
  kwiver::vital::config_block_key_t const&   key,
  kwiver::vital::config_block_value_t const& def )
{
  return self->get_value< kwiver::vital::config_block_value_t >( key, def );
}

pybind11::size_t
config_len( kv::config_block_sptr self )
{
  return self->available_values().size();
}

kv::config_block_value_t
config_getitem(
  kv::config_block_sptr self,
  kv::config_block_key_t const&  key )
{
  kv::config_block_value_t val;

  try
  {
    val = self->get_value< kv::config_block_value_t >( key );
  }
  catch( kv::no_such_configuration_value_exception const& )
  {
    std::ostringstream sstr;

    sstr << "\'" << key << "\'";

    PyErr_SetString( PyExc_KeyError, sstr.str().c_str() );
    throw py::error_already_set();
  }

  return val;
}

void
config_setitem(
  kv::config_block_sptr self,
  kv::config_block_key_t const& key,
  py::object const& value )
{
  kv::config_block_key_t const& str_value = py::str( value );

  self->set_value( key, str_value );
}

void
config_delitem(
  kv::config_block_sptr self,
  kv::config_block_key_t const& key )
{
  try
  {
    self->unset_value( key );
  }
  catch( kv::no_such_configuration_value_exception const& )
  {
    std::ostringstream sstr;

    sstr << "\'" << key << "\'";

    PyErr_SetString( PyExc_KeyError, sstr.str().c_str() );
    throw py::error_already_set();
  }
}

} // namespace python

} // namespace vital

} // namespace kwiver
