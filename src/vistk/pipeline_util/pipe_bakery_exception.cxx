/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "pipe_bakery_exception.h"

#include <sstream>

/**
 * \file pipe_bakery_exception.cxx
 *
 * \brief Implementations of exceptions used when baking a pipeline.
 */

namespace vistk
{

pipe_bakery_exception
::pipe_bakery_exception() throw()
  : pipeline_exception()
{
}

pipe_bakery_exception
::~pipe_bakery_exception() throw()
{
}

missing_cluster_block_exception
::missing_cluster_block_exception() throw()
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "A cluster block was not given when "
          "baking a cluster";

  m_what = sstr.str();
}

missing_cluster_block_exception
::~missing_cluster_block_exception() throw()
{
}

multiple_cluster_blocks_exception
::multiple_cluster_blocks_exception() throw()
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "Multiple cluster blocks were given "
          "when baking a cluster";

  m_what = sstr.str();
}

multiple_cluster_blocks_exception
::~multiple_cluster_blocks_exception() throw()
{
}

cluster_without_processes_exception
::cluster_without_processes_exception() throw()
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "A cluster cannot be baked without "
          "any processes";

  m_what = sstr.str();
}

cluster_without_processes_exception
::~cluster_without_processes_exception() throw()
{
}

cluster_without_ports_exception
::cluster_without_ports_exception() throw()
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "A cluster cannot be baked without "
          "any ports";

  m_what = sstr.str();
}

cluster_without_ports_exception
::~cluster_without_ports_exception() throw()
{
}

duplicate_cluster_port_exception
::duplicate_cluster_port_exception(process::port_t const& port, char const* const side) throw()
  : pipe_bakery_exception()
  , m_port(port)
{
  std::stringstream sstr;

  sstr << "The " << side << " port "
          "\'" << port << "\' was declared "
          "twice in a cluster";

  m_what = sstr.str();
}

duplicate_cluster_port_exception
::~duplicate_cluster_port_exception() throw()
{
}

duplicate_cluster_input_port_exception
::duplicate_cluster_input_port_exception(process::port_t const& port) throw()
  : duplicate_cluster_port_exception(port, "input")
{
}

duplicate_cluster_input_port_exception
::~duplicate_cluster_input_port_exception() throw()
{
}

duplicate_cluster_output_port_exception
::duplicate_cluster_output_port_exception(process::port_t const& port) throw()
  : duplicate_cluster_port_exception(port, "output")
{
}

duplicate_cluster_output_port_exception
::~duplicate_cluster_output_port_exception() throw()
{
}

unrecognized_config_flag_exception
::unrecognized_config_flag_exception(config::key_t const& key, config_flag_t const& flag) throw()
  : pipe_bakery_exception()
  , m_key(key)
  , m_flag(flag)
{
  std::stringstream sstr;

  sstr << "The \'" << m_key << "\' key "
          "has the \'" << m_flag << "\' on it "
          "which is unrecognized";

  m_what = sstr.str();
}

unrecognized_config_flag_exception
::~unrecognized_config_flag_exception() throw()
{
}

unrecognized_provider_exception
::unrecognized_provider_exception(config::key_t const& key, config_provider_t const& provider, config::value_t const& index) throw()
  : pipe_bakery_exception()
  , m_key(key)
  , m_provider(provider)
  , m_index(index)
{
  std::stringstream sstr;

  sstr << "The \'" << m_key << "\' key "
          "is requesting the index \'" << m_index << "\' "
          "from the unrecognized \'" << m_provider << "\'";

  m_what = sstr.str();
}

unrecognized_provider_exception
::~unrecognized_provider_exception() throw()
{
}

circular_config_provide_exception
::circular_config_provide_exception() throw()
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "There is a circular CONF provider request in the configuration";

  m_what = sstr.str();
}

circular_config_provide_exception
::~circular_config_provide_exception() throw()
{
}

unrecognized_system_index_exception
::unrecognized_system_index_exception(config::value_t const& index) throw()
  : pipe_bakery_exception()
  , m_index(index)
{
  std::stringstream sstr;

  sstr << "The \'" << m_index << "\' index "
          "does not exist for the SYS provider";

  m_what = sstr.str();
}

unrecognized_system_index_exception
::~unrecognized_system_index_exception() throw()
{
}

}
