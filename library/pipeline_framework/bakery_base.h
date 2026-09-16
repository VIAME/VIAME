// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   bakery_base.h
 * @brief  Interface for pipe bakery_base class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_BAKERY_BASE_H
#define SPROKIT_PIPELINE_UTIL_BAKERY_BASE_H

#include "pipe_declaration_types.h"

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/util/token_expander.h>
#include <viame/algorithm_framework/util/token_type_symtab.h>
#include <viame/algorithm_framework/logger/logger.h>

#include <map>
#include <variant>

namespace viame::pipeline {

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
    config_info_t( const viame::config_block_value_t& val,
                   bool ro,
                   bool relative_path,
                   const viame::source_location& loc );
    ~config_info_t();

    viame::config_block_value_t value;
    bool read_only;
    bool relative_path;
    viame::source_location defined_loc;
  };

  using config_decl_t = std::pair< viame::config_block_key_t, config_info_t >;
  using config_decls_t = std::vector< config_decl_t >;

  using process_decl_t = std::pair< process::name_t, process::type_t >;
  using process_decls_t =  std::vector< process_decl_t >;

  // The pipeline definition data items.
  config_decls_t m_configs;
  process_decls_t m_processes;
  process::connections_t m_connections;

  // Static methods
  static viame::config_block_key_t flatten_keys(viame::config_block_keys_t const& keys);
  static viame::config_block_sptr extract_configuration_from_decls( bakery_base::config_decls_t& configs );

  // static data
  static config_flag_t const flag_read_only;
  static config_flag_t const flag_tunable;
  static config_flag_t const flag_relativepath;
  static config_flag_t const flag_local_assign;

protected:
  void register_config_value( viame::config_block_key_t const&  root_key,
                              config_value_t const&                     value );

private:
  // macro provider
  std::shared_ptr< viame::token_expander > m_token_expander;
  viame::token_type_symtab* m_symtab;
  viame::config_block_sptr m_ref_config;

  viame::logger_handle_t m_logger;
};

} // namespace viame::pipeline

#endif /* SPROKIT_PIPELINE_UTIL_BAKERY_BASE_H */
