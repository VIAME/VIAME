/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#ifndef SPROKIT_TOOLS_BUILD_PIPELINE_FROM_OPTIONS_H
#define SPROKIT_TOOLS_BUILD_PIPELINE_FROM_OPTIONS_H

#include <sprokit/tools/sprokit_tools_export.h>

#include <sprokit/pipeline_util/pipeline_builder.h>
#include <boost/program_options/variables_map.hpp>

#include <boost/program_options/options_description.hpp>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Build pipeline from command line options
 *
 */
class SPROKIT_TOOLS_EXPORT build_pipeline_from_options
  : public pipeline_builder
{
public:

  build_pipeline_from_options();

  /**
   * \brief Create pipeline from command line input.
   *
   * This is the all-in-one call to create a pipeline builder.
   *
   * \param vm Variable map from parsing the command line
   * \param desc Command line options descriptions
   */
  build_pipeline_from_options( boost::program_options::variables_map const& vm,
                               boost::program_options::options_description const& desc );

  virtual ~build_pipeline_from_options() = default;

  /**
   * \brief Load options into builder.
   *
   * This method loads options as specified from the command
   * line. These options are supplementary config files and settings
   * as specified in the program options supplied.
   *
   * The result of this call is to add more entries to the internal
   * pipeline representation.
   *
   * \param vm Program options
   */
  void load_from_options( boost::program_options::variables_map const& vm );

}; // end class build_pipeline_from_options

} // end namespace

#endif // SPROKIT_TOOLS_BUILD_PIPELINE_FROM_OPTIONS_H
