/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#ifndef VITAL_TOOLS_EXPLORER_PLUGIN_H
#define VITAL_TOOLS_EXPLORER_PLUGIN_H

#include <vital/tools/explorer_plugin_export.h>

#include <vital/vital_config.h>
#include <vital/noncopyable.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/applets/cxxopts.hpp>
#include <vital/config/config_block.h>

namespace kwiver {
namespace vital {

class category_explorer;
using category_explorer_sptr = std::shared_ptr<category_explorer>;

// ----------------------------------------------------------------
/**
 * @brief Context for category_explorer.
 *
 * This class provides the interface to the plugin explorer that is
 * provided to the category_explorer plugin.
 */
class EXPLORER_PLUGIN_EXPORT explorer_context
  : private kwiver::vital::noncopyable
{
public:

  ~explorer_context();

  /**
   * @brief Get output stream.
   *
   * The stream returned is the current output stream for the
   * plugin_explorer took.
   *
   * @return Reference to the output stream.
   */
  std::ostream& output_stream() const;

  /**
   * @brief Collection of command line arguments.
   *
   * This method returns the active set of command line
   * options. Category_Explorer plugins can add any needed command
   * line arguments to this set.
   *
   * @return Pointer to command line args object.
   */
  cxxopts::Options& command_line_args();

  /**
   * @brief Parsed version of the command line
   *
   *
   * @return Reference to the parsed command line options.
   */
  cxxopts::ParseResult& command_line_result();


  /**
   * @brief Wrap long text to line length.
   *
   * The input string is wrapped to the line length used in the
   * tool. This is useful for formatting long description text blocks.
   *
   * @param text String to be wrapped.
   *
   * @return Input text with new-line characters added.
   */
  std::string wrap_text( const std::string& text ) const;

  /**
   * @brief Return formatting type string.
   *
   * This method returns the formatting type string that was specified
   * on the command line.
   *
   * @return Formatting type string.
   */
  const std::string& formatting_type() const;

  /**
   * @brief Display all factory attributes.
   *
   * This method displays all the attributes in the supplied
   * factory. This can be useful when the detail output option has
   * been selected so that all attributes can be displayed in raw
   * format in addition to this plugin's output format.
   */
  void display_attr( const kwiver::vital::plugin_factory_handle_t fact) const;

  /**
   * @brief Format description text.
   *
   * The supplied text is wrapped to the current formatting
   * specifications. The input text is in the standard description
   * format with a short description, followed by a blank line and
   * then the extended description. If the command is not operating in
   * the "detail" mode, then only the first line is returned.
   *
   * @param text Raw input text to be formatted
   *
   * @return Formatted description
   */
  std::string format_description( std::string const& text ) const;

  /**
   * @brief Print a config block.
   *
   * The specified config block is formatted and sent to the current
   * output stream. The formatting of the description is controlled by
   * the "detailed" option.
   *
   * @param config Block to format and print
   */
  void print_config( kwiver::vital::config_block_sptr const config ) const;

  bool if_detail() const;
  bool if_brief() const;

  class priv;

protected:
  explorer_context( priv * pp );

private:
  priv* p;

}; // end class explorer_context

// ----------------------------------------------------------------
/**
 * @brief Abstract base class for plugin category explorer.
 *
 * This class represents an extensible way of providing detailed
 * information about a category of plugins. The default exploration
 * process only displays the factory attributes. If the object created
 * by the factory has some interesting data to display, this explorer
 * plugin can do the introspection to extract all the details.
 *
 * Plugins are registered as follows. Note the specific name for the
 * registration function.
 *
\code
extern "C"
ALGO_EXPLORER_PLUGIN_EXPORT // Need appropriate export decorator
void register_explorer_plugin( kwiver::vital::plugin_loader& vpm )
{
  auto fact = vpm.ADD_FACTORY( kwiver::vital::category_explorer, kwiver::vital::algo_explorer );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Plugin explorer for algorithm category." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );
}
\endcode
 *
 * The name of the plugin specifies which plugin \b category it
 * handles. In this example, we are registering a plugin explorer for
 * those that are marked as algorithms.
 */
class category_explorer
{
public:
  // -- CONSTRUCTORS --
  category_explorer() = default;
  virtual ~category_explorer() = default;

  /**
   * @brief Initialize the plugin
   *
   * This method is called after this plugin is created to perform
   * initialization. Any initialization that is not done by the CTOR
   * should be done here. This is called in addition to the CTOR in
   * order to use the plugin factory for objects with no CTOR
   * parameters.
   *
   * @param context Reference to the tool context object.
   *
   * @return
   */
  virtual bool initialize( explorer_context* context ) = 0;

  /**
   * @brief Explore factory
   *
   * This method is called with the factory to explore.
   *
   * @param fact Factory to explore
   */
  virtual void explore( const kwiver::vital::plugin_factory_handle_t fact ) = 0;

}; // end class category_explorer

} } // end namespace

#endif /* VITAL_TOOLS_EXPLORER_PLUGIN_H */
