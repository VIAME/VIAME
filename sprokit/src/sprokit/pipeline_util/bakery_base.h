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
#include <vital/util/token_expander.h>
#include <vital/util/token_type_symtab.h>
#include <vital/logger/logger.h>
#include <vital/internal/variant/variant.hpp>

#include <map>

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief Base class for pipeline bakeries
 *
 * This class has the common behaviours for all the variant bakeries.
 */
class bakery_base
{
public:
  bakery_base();
  virtual ~bakery_base();

  void operator()( config_pipe_block const& config_block );
  void operator()( process_pipe_block const& process_block );
  void operator()( connect_pipe_block const& connect_block );


  /**
   * @brief Intermediate representation of a contfig entry.
   */
  class config_info_t
  {
  public:
    config_info_t( const kwiver::vital::config_block_value_t&           val,
                   bool                                  ro,
                   bool                                  relative_path,
                   const kwiver::vital::source_location& loc );
    ~config_info_t();

    kwiver::vital::config_block_value_t value;
    bool read_only;
    bool relative_path;
    kwiver::vital::source_location defined_loc;
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

  // static data
  static config_flag_t const flag_read_only;
  static config_flag_t const flag_tunable;
  static config_flag_t const flag_relativepath;
  static config_flag_t const flag_local_assign;

protected:
  void register_config_value( kwiver::vital::config_block_key_t const&  root_key,
                              config_value_t const&                     value );

private:
  // macro provider
  std::shared_ptr< kwiver::vital::token_expander > m_token_expander;
  kwiver::vital::token_type_symtab* m_symtab;
  kwiver::vital::config_block_sptr m_ref_config;

  kwiver::vital::logger_handle_t m_logger;
};

} // end namespace

#endif /* SPROKIT_PIPELINE_UTIL_BAKERY_BASE_H */
