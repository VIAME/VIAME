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

/**
 * @file   bakery_base.h
 * @brief  Interface for pipe bakery_base class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_BAKERY_BASE_H
#define SPROKIT_PIPELINE_UTIL_BAKERY_BASE_H

#include "pipe_declaration_types.h"

#include <vital/config/config_block.h>

#include <boost/variant.hpp>

#include <map>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Base class for pipeline bakeries
 *
 * This class has the common behaviours for all the variant bakeries.
 */
class bakery_base
  : public boost::static_visitor< >
{
public:
  bakery_base();
  virtual ~bakery_base();

  void operator()( config_pipe_block const& config_block );
  void operator()( process_pipe_block const& process_block );
  void operator()( connect_pipe_block const& connect_block );

  /**
   * \note We do *not* want std::map for the block management. With a map, we
   * may hide errors in the blocks (setting ro values, duplicate process
   * names, etc.)
   */

  typedef std::pair< config_provider_t, kwiver::vital::config_block_value_t > provider_request_t;
  typedef boost::variant< kwiver::vital::config_block_value_t, provider_request_t > config_reference_t;
  class config_info_t
  {
  public:
    typedef enum
    {
      append_none,
      append_string,
      append_comma,
      append_space,
      append_path
    } append_t;

    config_info_t( config_reference_t const&  ref,
                   bool                       ro,
                   append_t                   app );
    ~config_info_t();

    config_reference_t reference;
    bool read_only;
    append_t append;
  };
  typedef std::pair< kwiver::vital::config_block_key_t, config_info_t > config_decl_t;
  typedef std::vector< config_decl_t > config_decls_t;

  typedef std::pair< process::name_t, process::type_t > process_decl_t;
  typedef std::vector< process_decl_t > process_decls_t;

  config_decls_t m_configs;
  process_decls_t m_processes;
  process::connections_t m_connections;


  // Static methods
  static kwiver::vital::config_block_key_t flatten_keys(kwiver::vital::config_block_keys_t const& keys);
  static kwiver::vital::config_block_sptr extract_configuration_from_decls( bakery_base::config_decls_t& configs );
  static void dereference_static_providers( bakery_base::config_decls_t& bakery );


  // static data
  static config_flag_t const flag_read_only;
  static config_flag_t const flag_append;
  static config_flag_t const flag_append_prefix;
  static config_flag_t const flag_append_comma;
  static config_flag_t const flag_append_space;
  static config_flag_t const flag_append_path;
  static config_flag_t const flag_tunable;

  static config_provider_t const provider_config;
  static config_provider_t const provider_environment;
  static config_provider_t const provider_system;


protected:
  void register_config_value( kwiver::vital::config_block_key_t const&  root_key,
                              config_value_t const&                     value );
};

} // end namespace

#endif /* SPROKIT_PIPELINE_UTIL_BAKERY_BASE_H */
