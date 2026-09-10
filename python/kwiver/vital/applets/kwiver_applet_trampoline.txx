// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file kwiver_applet_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of kwiver_applet
 */

#ifndef KWIVER_APPLET_TRAMPOLINE_TXX
#define KWIVER_APPLET_TRAMPOLINE_TXX

#include <pybind11/pybind11.h>
#include <viame/algorithm_framework/applets/kwiver_applet.h>
#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver {

namespace tools {

namespace python {

template < class applet_base = kwiver::tools::kwiver_applet >
class kwiver_applet_trampoline : public applet_base
{
public:
  using applet_base::applet_base;

  // Pure virtual function that must be overridden
  int
  run() override
  {
    PYBIND11_OVERLOAD_PURE(
      int,
      applet_base,
      run,
    );
  }

  // Virtual functions with default implementations
  void
  add_command_options() override
  {
    PYBIND11_OVERLOAD(
      void,
      applet_base,
      add_command_options,
    );
  }

  void
  set_configuration( kwiver::vital::config_block_sptr cb ) override
  {
    PYBIND11_OVERLOAD(
      void,
      applet_base,
      set_configuration,
      cb
    );
  }

  kwiver::vital::config_block_sptr
  get_configuration() const override
  {
    PYBIND11_OVERLOAD(
      kwiver::vital::config_block_sptr,
      applet_base,
      get_configuration,
    );
  }

  void
  initialize() override
  {
    PYBIND11_OVERLOAD(
      void,
      applet_base,
      initialize,
    );
  }

  void
  set_configuration_internal( kwiver::vital::config_block_sptr cb ) override
  {
    PYBIND11_OVERLOAD(
      void,
      applet_base,
      set_configuration_internal,
      cb
    );
  }

  // Public accessors for protected methods
  const std::string&
  public_applet_name() const
  {
    return applet_base::applet_name();
  }

  const std::vector< std::string >&
  public_applet_args() const
  {
    return applet_base::applet_args();
  }

  std::string
  public_wrap_text( const std::string& text )
  {
    return applet_base::wrap_text( text );
  }
};

} // namespace python

} // namespace tools

} // namespace kwiver

#endif
