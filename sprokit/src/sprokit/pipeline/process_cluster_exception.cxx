// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
