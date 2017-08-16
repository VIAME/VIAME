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

#include "pipeline_builder.h"

#include <sprokit/pipeline_util/load_pipe.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_declaration_types.h>
#include <sprokit/pipeline/pipeline.h>

#include <vital/config/config_block.h>
#include <vital/util/tokenize.h>

#include <boost/filesystem/operations.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace sprokit
{

namespace
{

static std::string const split_str = "=";

}

// ==================================================================
pipeline_builder
::pipeline_builder()
  : m_blocks()
{
}


// ------------------------------------------------------------------
void
pipeline_builder
::load_pipeline(std::istream& istr, std::string const& def_file )
{
  m_blocks = sprokit::load_pipe_blocks(istr, def_file);
}


// ------------------------------------------------------------------
void
pipeline_builder
::load_supplement( kwiver::vital::path_t const& path)
{
  sprokit::pipe_blocks const supplement = sprokit::load_pipe_blocks_from_file(path);

  m_blocks.insert(m_blocks.end(), supplement.begin(), supplement.end());
}


// ------------------------------------------------------------------
void
pipeline_builder
::add_setting(std::string const& setting)
{
  sprokit::config_pipe_block block;

  sprokit::config_value_t value;

  size_t const split_pos = setting.find(split_str);

  if (split_pos == std::string::npos)
  {
    std::string const reason = "Error: The setting on the command line "
                               "\'" + setting + "\' does not contain "
                               "the \'" + split_str + "\' string which "
                               "separates the key from the value";

    throw std::runtime_error(reason);
  }

  kwiver::vital::config_block_key_t setting_key = setting.substr(0, split_pos);
  kwiver::vital::config_block_value_t setting_value = setting.substr(split_pos + split_str.size());

  kwiver::vital::config_block_keys_t keys;

  kwiver::vital::tokenize( setting_key, keys, kwiver::vital::config_block::block_sep, kwiver::vital::TokenizeTrimEmpty );

  if (keys.size() < 2)
  {
    std::string const reason = "Error: The key portion of setting "
                               "\'" + setting + "\' does not contain "
                               "at least two keys in its keypath which is "
                               "invalid. (e.g. must be at least a:b)";

    throw std::runtime_error(reason);
  }

  value.key_path.push_back(keys.back());
  value.value = setting_value;

  keys.pop_back();

  block.key = keys;
  block.values.push_back(value);

  m_blocks.push_back(block);
}


// ------------------------------------------------------------------
sprokit::pipeline_t
pipeline_builder
::pipeline() const
{
  return sprokit::bake_pipe_blocks(m_blocks);
}


// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
pipeline_builder
::config() const
{
  return sprokit::extract_configuration(m_blocks);
}


// ------------------------------------------------------------------
sprokit::pipe_blocks
pipeline_builder
::blocks() const
{
  return m_blocks;
}

}
