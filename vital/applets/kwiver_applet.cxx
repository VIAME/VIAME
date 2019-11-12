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

#include "kwiver_applet.h"

#include "applet_context.h"

namespace kwiver {
namespace tools {

// ----------------------------------------------------------------------------
kwiver_applet::
kwiver_applet()
{
}


kwiver_applet::
~kwiver_applet()
{
}


// ----------------------------------------------------------------------------
void
kwiver_applet::
add_command_options()
{
  // The default implementation assumes that the applet will do its
  // own arg parsing if it hasn't registered any specific args for the
  // tool_runner to parse.
  if ( m_context )
  {
    m_context->m_skip_command_args_parsing = true;
  }
  else
  {
    throw std::runtime_error( "Invalid context pointer" );
  }

}


// ----------------------------------------------------------------------------
cxxopts::ParseResult&
kwiver_applet::
command_args()
{
  if (m_context && m_context->m_result)
  {
    return *m_context->m_result;
  }

  throw std::runtime_error( "Invalid context pointer or command line results are not available." );
}

// ----------------------------------------------------------------------------
void
kwiver_applet::
initialize( kwiver::tools::applet_context* ctxt)
{
  m_context = ctxt;
  m_cmd_options.reset( new cxxopts::Options( applet_name(), "" ) );
}

// ----------------------------------------------------------------------------
std::string
kwiver_applet::
wrap_text( const std::string& text )
{
  if ( m_context )
  {
    return m_context->m_wtb.wrap_text( text );
  }

  throw std::runtime_error( "Invalid context pointer" );
}

// ----------------------------------------------------------------------------
const std::vector<std::string>&
kwiver_applet::
applet_args() const
{
  if ( m_context )
  {
    return m_context->m_argv;
  }

  throw std::runtime_error( "Invalid context pointer" );
}

// ----------------------------------------------------------------------------
const std::string&
kwiver_applet::
applet_name() const
{
  if ( m_context )
  {
    return m_context->m_applet_name;
  }

  throw std::runtime_error( "Invalid context pointer" );
}

} } // end namespace kwiver
