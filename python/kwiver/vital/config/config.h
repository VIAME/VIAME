// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PYTHON_CONFIG_H
#define KWIVER_VITAL_PYTHON_CONFIG_H

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/config/config_difference.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace viame {

namespace python {

/// Cast python object to a configuration block value
viame::config_block_value_t
config_block_set_value_cast( py::object const& value );

/// Set a value in configuration block using key and value
void
config_set_value(
  viame::config_block_sptr self,
  viame::config_block_key_t const& key,
  viame::config_block_key_t const& value );

/// Get a value in a configuration block using key
viame::config_block_value_t
config_get_value(
  viame::config_block_sptr self,
  viame::config_block_key_t const& key );

/// Get a value in a configuration block using key and a default value
viame::config_block_value_t
config_get_value_with_default(
  viame::config_block_sptr self,
  viame::config_block_key_t const& key,
  viame::config_block_value_t const& def );

/// Determine the number of elements in config block
py::size_t
config_len( viame::config_block_sptr self );

/// Get a value in a configuration block using key
viame::config_block_value_t
config_getitem(
  viame::config_block_sptr self,
  viame::config_block_key_t const& key );

/// Set a value in a configuration block using key and value
void
config_setitem(
  viame::config_block_sptr self,
  viame::config_block_key_t const& key,
  py::object const& value );

/// Delete value in a configuration block using a key
void
config_delitem(
  viame::config_block_sptr self,
  viame::config_block_key_t const& key );

/// pybind11 module for config
void
config( py::module& m );

} // namespace python

} // namespace viame

#endif
