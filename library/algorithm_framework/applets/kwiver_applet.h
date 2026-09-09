// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOLS_KWIVER_APPLET_H
#define KWIVER_TOOLS_KWIVER_APPLET_H

#include <viame/algorithm_framework/applets/vital_applets_export.h>

#include <cxxopts.hpp>
#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/pluggable.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>
#include <viame/algorithm_framework/plugin/plugin_factory.h>
#include <viame/algorithm_framework/plugin/plugin_info.h>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace kwiver {

namespace tools {

// forward type definition
class applet_context;

/// @brief Abstract base class for all kwiver tools.
///
/// This class represents the abstract base class for all loadable
/// applets.
class VITAL_APPLETS_EXPORT kwiver_applet : public vital::pluggable
{
public:
  kwiver_applet();
  PLUGGABLE_INTERFACE( kwiver_applet );

  void initialize( kwiver::tools::applet_context* ctxt );

  /// @brief Main part of the applet.
  ///
  /// This method implements the main functionality of the applet. This
  /// is called for the applet to do its stuff.
  ///
  /// @return Application return code.
  virtual int run() = 0;

  /// @brief find and read a config file on the KWIVER config path
  ///
  /// Searches for a configuration file with the given file name in the current
  /// directory and on the KWIVER config search path relative to the kwiver
  /// executable location.
  static
  kwiver::vital::config_block_sptr
  find_configuration( std::string const& file_name );

  /// @brief Add command line options to parser.
  ///
  /// This method adds the program description and command line options
  /// to the command line parser. Command line processing will be skipped
  /// if this method is not overridden.
  ///
  /// Command line specification is added directly to this->m_cmd_options.
  ///
  /// Positional arguments
  /// \code
  /// m_cmd_options.add_options()
  /// ("input", "Input file", cxxopts::value<std::string>())
  /// ("output", "Output file", cxxopts::value<std::string>())
  /// ("positional", "Positional parameters",
  ///  cxxopts::value<std::vector<std::string>>())
  /// ;
///
  /// m_cmd_options.parse_positional({"input", "output", "positional"});
///
  /// \endcode
  ///
  /// Adding command option groups
  ///
/// \code
/// m_cmd_options.add_option("group")
/// ( "I,path", "Add directory search path")
/// ;
///
/// \endcode
  virtual void add_command_options();

  /// @brief Return ref to parse results
  ///
  /// This method returns a reference to the command line parser
  /// results.
  ///
  /// @return Ref to parser results.
  cxxopts::ParseResult& command_args();

  /// Command line options specification. This is initialized by the
  /// add_command_options() method as delegated to the derived applet.
  /// This is managed by unique pointer to delay creation.
  std::unique_ptr< cxxopts::Options > m_cmd_options;

  /// Set this applet's properties via a config block
  ///
  /// This method is called to pass a configuration to the
  /// applet. The implementation of this method should be
  /// light-weight and only save the necessary config values. Defer
  /// any substantial processing in another method.
  ///
  /// \throws no_such_configuration_value_exception
  ///    Thrown if an expected configuration value is not present.
  ///
  /// \throws algorithm_configuration_exception
  ///    Thrown when the algorithm is given an invalid \c config_block or is
  ///    otherwise unable to configure itself.
  ///
  /// \param config  The \c config_block instance containing the configuration
  ///                parameters for this applet
  virtual void set_configuration(
    [[maybe_unused]] vital::config_block_sptr cb ) {}

  /// Get this applet's \link kwiver::vital::config_block configuration
  /// block \endlink
  ///
  /// This method returns the required configuration for the
  /// applet. The implementation of this method should be
  /// light-weight and only create and fill in the config
  /// block.
  ///
  /// This base virtual function implementation returns an empty configuration.
  ///
  /// \returns \c config_block containing the configuration for this applet
  ///          and any nested components.
  virtual vital::config_block_sptr get_configuration() const;

protected:
  /// @brief Get applet name
  ///
  /// This method returns the name of the applit as it was specified on
  /// the command line.
  ///
  /// @return Applet name
  const std::string& applet_name() const;

  /// @brief Wrap text block.
  ///
  /// This method wraps the supplied text into a fixed width text
  /// block.
  ///
  /// @param text Input text string to be wrapped.
  ///
  /// @return Text string wrapped into a block.
  std::string wrap_text( const std::string& text );

  /// @brief Return original arguments
  ///
  /// The vector of original applet args is returned.
  ///
  /// @return Read only vector of args
  const std::vector< std::string >& applet_args() const;

  // \brief Initialize the internals of the applet.
  //
  // This is overridden every time an applet needs to initialize any internal
  // state. The pluggable macros will make sure to call it in auto-generated
  // constructor.
  virtual void initialize() {}

  // \brief Run additional logic duting set_configuration.
  //
  // PLUGGABLE_IMPL will autogenerate a default implemention for
  // set_configutation. If however there is a need to execute adiitional logic
  // after the member variable have been set this fuction should be overidden
  // to hold that logic.
  virtual void set_configuration_internal(
    [[maybe_unused]] vital::config_block_sptr cb ) {}

private:
  /// Context provided by the applet runner.
  kwiver::tools::applet_context* m_context { nullptr };
};

typedef std::shared_ptr< kwiver_applet > kwiver_applet_sptr;

} // namespace tools

namespace vital {

/// Simple factory for applets that use zero-argument construction
///
/// Applets don't use the config block system for construction,
/// they use command-line arguments instead. This factory provides
/// a simple way to create applet instances.
template < class APPLET >
class applet_plugin_factory
  : public plugin_factory
{
public:
  static_assert(
    std::is_base_of< kwiver::tools::kwiver_applet, APPLET >::value,
    "The given applet type must derive from kwiver_applet." );

  explicit applet_plugin_factory()
    : plugin_factory( typeid( kwiver::tools::kwiver_applet ).name() )
  {
    this->add_attribute( INTERFACE_TYPE, "kwiver_applet" )
      .add_attribute( CONCRETE_TYPE, typeid( APPLET ).name() );
  }

  pluggable_sptr
  from_config( [[maybe_unused]] config_block_sptr const cb ) const override
  {
    return std::make_shared< APPLET >();
  }

  void
  get_default_config( [[maybe_unused]] config_block& cb ) const override
  {
    // Applets don't use config blocks for construction
  }

  ~applet_plugin_factory() override = default;
};

} // namespace vital

}   // end namespace

// ----------------------------------------------------------------------------
// Support for adding factories

#define ADD_APPLET( applet_T ) \
add_factory(                   \
  new kwiver::vital::applet_plugin_factory< applet_T >() )

#endif // KWIVER_TOOLS_KWIVER_APPLET_H
