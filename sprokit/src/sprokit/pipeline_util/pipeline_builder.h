/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#ifndef SPROKIT_TOOLS_PIPELINE_BUILDER_H
#define SPROKIT_TOOLS_PIPELINE_BUILDER_H

#include "pipeline_util-config.h"

#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline/types.h>

#include <vital/vital_types.h>
#include <vital/noncopyable.h>

#include <istream>
#include <string>
#include <memory>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * \brief Class to build pipeline.
 *
 * This class is the main interface for creating a pipeline. A
 * pipeline builder (object of this class) is created and configured
 * with the pipeline description. Configuration items can be added as
 * needed.
 *
 * After the pipeline builder is configured as desired, it can build
 * the pipeline which is ready to process.
 */
class SPROKIT_PIPELINE_UTIL_EXPORT pipeline_builder
  : kwiver::vital::noncopyable
{
public:
  /**
   * \brief Create default pipeline builder object.
   *
   */
  pipeline_builder();
  virtual ~pipeline_builder() = default;

  /**
   * \brief Load pipeline configuration from stream.
   *
   * This method loads the pipeline configuration into the
   * builder. This can be used to add additional configuration files
   * to the internal pipeline representation.
   *
   * \param istr Stream containing the textual pipeline definition.
   * \param def_file The default file name used when reporting errors
   * from the stream. The directory portion is used when resolving
   * included files and relpath specifiers.
   */
  void load_pipeline(std::istream& istr, std::string const& def_file = "" );

  /**
   * \brief Load supplemental data into pipeline description.
   *
   * Adds supplemental block to the internal representation of the pipeline.
   *
   * \param path File to read.
   */
  void load_supplement( kwiver::vital::path_t const& path);

  /**
   * \brief Add single config entry
   *
   * Add a single config entry to the internal pipeline
   * representation. A config entry has for form key=value
   *
   * \param setting String containing a single config setting entry.
   */
  void add_setting(std::string const& setting);

  /**
   * \brief Create pipeline from internal representation.
   *
   * This method instantiates a pipeline from the accumulated internal
   * pipeline representation.
   *
   * \return A new pipeline object.
   */
  sprokit::pipeline_t pipeline() const;

  /**
   * \brief Extract config block from pipeline.
   *
   * This method extracts the config for the pipeline.
   *
   * \return Block containing the whole pipeline config.
   */
  kwiver::vital::config_block_sptr config() const;

  /**
   * \brief Get internal representation of pipeline.
   *
   *
   * \return List of internal pipeline blocks.
   */
  sprokit::pipe_blocks blocks() const;

private:
  sprokit::pipe_blocks m_blocks;
};

}

#endif // SPROKIT_TOOLS_PIPELINE_BUILDER_H
