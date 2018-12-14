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

#include "kwiver_tools_applet_export.h"

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
 */
class KWIVER_TOOLS_APPLET_EXPORT kwiver_applet
{
public:
  kwiver_applet();
  virtual ~kwiver_applet();

  void initialize( kwiver::tools::applet_context* ctxt);

  virtual int run( const std::vector<std::string>& argv ) = 0;
  virtual void usage( std::ostream& outstream ) const = 0;

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

private:
  kwiver::tools::applet_context* m_context;
};

typedef std::shared_ptr<kwiver_applet> kwiver_applet_sptr;

} } // end namespace

// ==================================================================
// Support for adding factories

#define ADD_APPLET( applet_T)                               \
  add_factory( new kwiver::vital::plugin_factory_0< applet_T >( typeid( kwiver::tools::kwiver_applet ).name() ) )

#endif /* KWIVER_TOOLS_KWIVER_APPLET_H */
