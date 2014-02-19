/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
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

#include "tools-config.h"

#include <sprokit/pipeline_util/path.h>
#include <sprokit/pipeline_util/pipe_bakery.h>

#include <sprokit/pipeline/types.h>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/noncopyable.hpp>

#include <istream>
#include <string>

namespace sprokit
{

class SPROKIT_TOOLS_EXPORT pipeline_builder
  : boost::noncopyable
{
  public:
    pipeline_builder(boost::program_options::variables_map const& vm, boost::program_options::options_description const& desc);
    pipeline_builder();
    ~pipeline_builder();

    void load_pipeline(std::istream& istr);

    void load_from_options(boost::program_options::variables_map const& vm);

    void load_supplement(sprokit::path_t const& path);
    void add_setting(std::string const& setting);

    sprokit::pipeline_t pipeline() const;
    sprokit::config_t config() const;
    sprokit::pipe_blocks blocks() const;
  private:
    sprokit::pipe_blocks m_blocks;
};

SPROKIT_TOOLS_EXPORT boost::program_options::options_description pipeline_common_options();
SPROKIT_TOOLS_EXPORT boost::program_options::options_description pipeline_input_options();
SPROKIT_TOOLS_EXPORT boost::program_options::options_description pipeline_output_options();
SPROKIT_TOOLS_EXPORT boost::program_options::options_description pipeline_run_options();

}

#endif // SPROKIT_TOOLS_PIPELINE_BUILDER_H
