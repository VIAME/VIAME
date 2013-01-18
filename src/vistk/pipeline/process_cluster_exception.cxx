/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process_cluster_exception.h"

#include <sstream>

/**
 * \file process_cluster_exception.cxx
 *
 * \brief Implementation of exceptions used within \link vistk::process_clsuter process clusters\endlink.
 */

namespace vistk
{

mapping_after_process_exception
::mapping_after_process_exception(process::name_t const& name, config::key_t const& key, process::name_t const& mapped_name, config::key_t const& mapped_key) throw()
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
          "but it has already been created";

  m_what = sstr.str();
}

mapping_after_process_exception
::~mapping_after_process_exception() throw()
{
}

}
