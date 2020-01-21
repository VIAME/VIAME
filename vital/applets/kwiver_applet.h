/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#ifndef KWIVER_TOOLS_KWIVER_APPLET_H
#define KWIVER_TOOLS_KWIVER_APPLET_H

#include <vital/applets/kwiver_tools_applet_export.h>
#include <vital/plugin_loader/plugin_info.h>

#include <vital/applets/cxxopts.hpp>

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

  virtual int run() = 0;

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
