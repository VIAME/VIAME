/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

#include "plugin_paths.h"

#include <vital/util/tokenize.h>

#include <kwiversys/SystemTools.hxx>

namespace sprokit {

namespace {  // anonymous

static std::string const default_module_paths = std::string( DEFAULT_MODULE_PATHS );

static char const* environment_variable_name( "SPROKIT_MODULE_PATH" );
  // SPROKIT_PLUGIN_PATH
  // SPROKIT_CLUSTER_PATH

} // end namespace


// ------------------------------------------------------------------
std::vector< std::string >
plugin_paths()
{
  std::vector< std::string > ret_val;

  // Check env variable for path specification
  const char * env_ptr = kwiversys::SystemTools::GetEnv( environment_variable_name );
  if ( 0 != env_ptr )
  {
    // LOG_DEBUG( m_priv->m_logger, "Adding path(s) \"" << env_ptr << "\" from environment" );
    std::string const extra_module_dirs(env_ptr);

    // Split supplied path into separate items using PATH_SEPARATOR_CHAR as delimiter
    kwiver::vital::tokenize( extra_module_dirs, ret_val, PATH_SEPARATOR_CHAR, true );
  }

  // Add default paths
  kwiver::vital::tokenize( default_module_paths, ret_val, PATH_SEPARATOR_CHAR, true );

  return ret_val;
}

} // end namespace
