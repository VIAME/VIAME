// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef PYTHON_PLUGIN_FACTORY_H
#define PYTHON_PLUGIN_FACTORY_H

#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/plugin_factory.h>

#include <memory>

namespace py = pybind11;

namespace viame::python {

/// @brief Factory to register and generate python instances for an interface.
class python_plugin_factory
  : public plugin_factory
{
public:
  explicit python_plugin_factory( py::object const& python_type )
    : m_python_type( python_type )
  {
    // Get the plugin name - prefer plugin_name() if available, fall back to
    // __name__
    std::string plugin_name;
    if( py::hasattr( python_type, "plugin_name" ) )
    {
      plugin_name = python_type.attr( "plugin_name" )().cast< std::string > ();
    }
    else
    {
      plugin_name = python_type.attr( "__name__" ).cast< std::string >();
    }

    this->add_attribute( plugin_factory::INTERFACE_TYPE,
      python_type.attr( "interface_name" )()
        .cast< std::string > () )
      .add_attribute( plugin_factory::CONCRETE_TYPE,
                      python_type.attr( "__name__" ).cast< std::string > () )
        .add_attribute( plugin_factory::PLUGIN_NAME, plugin_name );
  }

  ~python_plugin_factory() override = default;

  pluggable_sptr
  from_config( const config_block_sptr cb ) const override
  {
    py::gil_scoped_acquire gil;
    py::object instance = m_python_type.attr( "from_config" )( cb );
    return instance.cast< pluggable_sptr >();
  }

  void
  get_default_config( config_block& cb ) const override
  {
    py::gil_scoped_acquire gil;

    // Hand python a `config_block_sptr`, not the reference.
    //
    // `config_block` is bound as `py::class_< config_block, config_block_sptr >`,
    // so pybind11 holds it by shared pointer and cannot make a python object
    // out of a bare reference without copying one -- and `config_block` is
    // non-copyable. Passing `cb` straight through therefore threw
    // "return_value_policy = copy, but type config_block is non-copyable"
    // for **every python implementation**, which `registry-dump` recorded as
    // an error and `compare_registry.py` then skipped. The effect was that
    // an algorithm ported from C++ to python silently stopped having its
    // config keys and defaults checked against the baseline.
    //
    // Both callers own the block through a `config_block_sptr` already, so
    // `shared_from_this` gives the real owner. The fallback is for a caller
    // that does not: a non-owning handle, which is safe because the callee
    // only writes keys into it and the pointer does not outlive this call.
    config_block_sptr handle;

    try
    {
      handle = cb.shared_from_this();
    }
    catch( std::bad_weak_ptr const& )
    {
      handle = config_block_sptr( &cb, []( config_block* ){} );
    }

    m_python_type.attr( "get_default_config" )( handle );
  }

private:
  py::object m_python_type;
};

} // namespace viame::python

#endif // PYTHON_PLUGIN_FACTORY_H
