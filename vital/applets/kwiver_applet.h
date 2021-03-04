// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOLS_KWIVER_APPLET_H
#define KWIVER_TOOLS_KWIVER_APPLET_H

#include <vital/applets/kwiver_tools_applet_export.h>
#include <vital/plugin_loader/plugin_info.h>

#include <vital/applets/cxxopts.hpp>
#include <vital/config/config_block.h>

#include <ostream>
#include <memory>
#include <vector>
#include <string>

namespace kwiver {
namespace tools {

// forward type definition
class applet_context;

/**
 * @brief Abstract base class for all kwiver tools.
 *
 * This class represents the abstract base class for all loadable
 * applets.
 */
class KWIVER_TOOLS_APPLET_EXPORT kwiver_applet
{
public:
  kwiver_applet();
  virtual ~kwiver_applet();

  void initialize( kwiver::tools::applet_context* ctxt );

  /**
   * @brief Main part of the applet.
   *
   * This method implements the main functionality of the applet. This
   * is called for the applet to do its stuff.
   *
   * @return Application return code.
   */
  virtual int run() = 0;

  /**
   * @brief find and read a config file on the KWIVER config path
   *
   * Searches for a configuration file with the given file name in the current
   * directory and on the KWIVER config search path relative to the kwiver
   * executable location.
   */
  static
  kwiver::vital::config_block_sptr
    find_configuration(std::string const& file_name );

  /**
   * @brief Add command line options to parser.
   *
   * This method adds the program description and command line options
   * to the command line parser. Command line processing will be skipped
   * if this method is not overridden.
   *
   * Command line specification is added directly to this->m_cmd_options.
   *
   * Positional arguments
   \code
   m_cmd_options.add_options()
    ("input", "Input file", cxxopts::value<std::string>())
    ("output", "Output file", cxxopts::value<std::string>())
    ("positional", "Positional parameters",
      cxxopts::value<std::vector<std::string>>())
  ;

  m_cmd_options.parse_positional({"input", "output", "positional"});

  \endcode
  *
  * Adding command option groups
  *
  *
\code
m_cmd_options.add_option("group")
( "I,path", "Add directory search path")
;

\endcode
  */
  virtual void add_command_options();

  /**
   * @brief Return ref to parse results
   *
   * This method returns a reference to the command line parser
   * results.
   *
   * @return Ref to parser results.
   */
  cxxopts::ParseResult& command_args();

  /**
   * Command line options specification. This is initialized by the
   * add_command_options() method as delegated to the derived applet.
   * This is managed by unique pointer to delay creation.
   */
  std::unique_ptr< cxxopts::Options > m_cmd_options;

protected:

  /**
   * @brief Get applet name
   *
   * This method returns the name of the applit as it was specified on
   * the command line.
   *
   * @return Applet name
   */
  const std::string& applet_name() const;

  /**
   * @brief Wrap text block.
   *
   * This method wraps the supplied text into a fixed width text
   * block.
   *
   * @param text Input text string to be wrapped.
   *
   * @return Text string wrapped into a block.
   */
  std::string wrap_text( const std::string& text );

  /**
   * @brief Return original arguments
   *
   * The vector of original applet args is returned.
   *
   * @return Read only vector of args
   */
  const std::vector<std::string>& applet_args() const;

private:
  /**
   * Context provided by the applet runner.
   */
  kwiver::tools::applet_context* m_context {nullptr};

};

typedef std::shared_ptr<kwiver_applet> kwiver_applet_sptr;

} } // end namespace

// ==================================================================
// Support for adding factories

#define ADD_APPLET( applet_T)                               \
  add_factory( new kwiver::vital::plugin_factory_0< applet_T >( typeid( kwiver::tools::kwiver_applet ).name() ) )

#endif /* KWIVER_TOOLS_KWIVER_APPLET_H */
