// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef PYTHON_PLUGIN_FACTORY_H
#define PYTHON_PLUGIN_FACTORY_H

#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/plugin/plugin_factory.h>

namespace py = pybind11;

namespace kwiver::vital::python {

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
    m_python_type.attr( "get_default_config" )( cb );
  }

private:
  py::object m_python_type;
};

} // namespace

#endif // PYTHON_PLUGIN_FACTORY_H
