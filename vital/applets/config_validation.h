/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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

#ifndef KWIVER_TOOLS_CONFIG_VALIDATION_H
#define KWIVER_TOOLS_CONFIG_VALIDATION_H

#include <vital/applets/kwiver_tools_applet_export.h>

#include <vital/config/config_block.h>
#include <vital/logger/logger.h>

#include <string>

namespace kwiver {
namespace tools {

KWIVER_TOOLS_APPLET_EXPORT
bool
validate_required_input_file(std::string const& name,
                             kwiver::vital::config_block const& config,
                             kwiver::vital::logger_handle_t logger);

KWIVER_TOOLS_APPLET_EXPORT
bool
validate_optional_input_file(std::string const& name,
                             kwiver::vital::config_block const& config,
                             kwiver::vital::logger_handle_t logger);

KWIVER_TOOLS_APPLET_EXPORT
bool
validate_required_output_file(std::string const& name,
                              kwiver::vital::config_block const& config,
                              kwiver::vital::logger_handle_t logger,
                              bool make_directory = true,
                              bool test_write = true);

KWIVER_TOOLS_APPLET_EXPORT
bool
validate_optional_output_file(std::string const& name,
                              kwiver::vital::config_block const& config,
                              kwiver::vital::logger_handle_t logger,
                              bool make_directory = true,
                              bool test_write = true);

KWIVER_TOOLS_APPLET_EXPORT
bool
validate_required_output_dir(std::string const& name,
                             kwiver::vital::config_block const& config,
                             kwiver::vital::logger_handle_t logger,
                             bool make_directory = true);

KWIVER_TOOLS_APPLET_EXPORT
bool
validate_optional_output_dir(std::string const& name,
                             kwiver::vital::config_block const& config,
                             kwiver::vital::logger_handle_t logger,
                             bool make_directory = true);

}
}
#endif
