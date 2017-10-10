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

#include "process_cluster_exception.h"

#include <sstream>

/**
 * \file process_cluster_exception.cxx
 *
 * \brief Implementation of exceptions used within \link sprokit::process_clsuter process clusters\endlink.
 */

namespace sprokit
{

process_cluster_exception
::process_cluster_exception() noexcept
  : process_exception()
{
}

process_cluster_exception
::~process_cluster_exception() noexcept
{
}

mapping_after_process_exception
::mapping_after_process_exception(process::name_t const& name,
                                  kwiver::vital::config_block_key_t const& key,
                                  process::name_t const& mapped_name,
                                  kwiver::vital::config_block_key_t const& mapped_key) noexcept
  : process_cluster_exception()
  , m_name(name)
  , m_key(key)
  , m_mapped_name(mapped_name)
  , m_mapped_key(mapped_key)
{
  std::ostringstream sstr;

  sstr << "The \'" << m_key << "\' configuration on "
          "the process cluster \'" << m_name << "\' was "
          "requested for mapping to the \'" << m_mapped_key << "\' "
          "configuration on the process \'" << m_name << "\', "
          "but the process has already been created";

  m_what = sstr.str();
}

mapping_after_process_exception
::~mapping_after_process_exception() noexcept
{
}

mapping_to_read_only_value_exception
::mapping_to_read_only_value_exception(process::name_t const& name,
                                       kwiver::vital::config_block_key_t const& key,
                                       kwiver::vital::config_block_value_t const& value,
                                       process::name_t const& mapped_name,
                                       kwiver::vital::config_block_key_t const& mapped_key,
                                       kwiver::vital::config_block_value_t const& ro_value) noexcept
  : process_cluster_exception()
  , m_name(name)
  , m_key(key)
  , m_value(value)
  , m_mapped_name(mapped_name)
  , m_mapped_key(mapped_key)
  , m_ro_value(ro_value)
{
  std::ostringstream sstr;

  sstr << "The \'" << m_key << "\' configuration on "
          "the process cluster \'" << m_name << "\' was "
          "requested for mapping to the \'" << m_mapped_key << "\' "
          "configuration on the process \'" << m_name << "\', "
          "but it was given as a read-only value "
          "\'" << m_ro_value << "\' and the mapping is set to "
          "\'" << m_value << "\'";

  m_what = sstr.str();
}

mapping_to_read_only_value_exception
::~mapping_to_read_only_value_exception() noexcept
{
}

}
