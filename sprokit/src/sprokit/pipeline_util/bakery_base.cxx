/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
 * @file   bakery_base.cxx
 * @brief  Implementation for bakery_base class
 */

#include "bakery_base.h"

#include "pipe_bakery_exception.h"

#include <vital/util/string.h>
#include <vital/util/token_type_sysenv.h>
#include <vital/util/token_type_env.h>
#include <vital/config/token_type_config.h>

#include <kwiversys/SystemTools.hxx>

namespace sprokit {

namespace {

class expander_bakery
  : public kwiver::vital::token_expander
{
public:
  expander_bakery( kwiver::vital::logger_handle_t logger)
    : kwiver::vital::token_expander()
    , m_logger( logger )
  { }

protected:
  virtual bool handle_missing_entry( const std::string& provider, const std::string& entry )
  {
    std::stringstream str;
    str <<  "Entry for provider \"" << provider << "\" does not have an element \""
        << entry << "\"";
    VITAL_THROW( provider_error_exception, str.str() );

    // Could do a log message instead
    return true;
  }


  virtual bool handle_missing_provider( const std::string& provider, const std::string& entry )
  {
    std::stringstream str;
    str << "Provider \"" << provider << "\" is not available";
    VITAL_THROW( provider_error_exception, str.str() );

    // Could do a log message instead
    return true;
  }

private:
    kwiver::vital::logger_handle_t m_logger;
};

} // end namespace

config_flag_t const bakery_base::flag_read_only = config_flag_t("ro");
config_flag_t const bakery_base::flag_tunable = config_flag_t("tunable");
config_flag_t const bakery_base::flag_relativepath = config_flag_t("relativepath");
config_flag_t const bakery_base::flag_local_assign = config_flag_t("local-assign");

// ------------------------------------------------------------------
bakery_base
::bakery_base()
  : m_configs()
  , m_processes()
  , m_connections()
  , m_symtab( new kwiver::vital::token_type_symtab("LOCAL") )
  , m_ref_config( kwiver::vital::config_block::empty_config() )
  , m_logger( kwiver::vital::get_logger( "sprokit.bakery_base" ) )
{
  m_token_expander = std::make_shared < expander_bakery >(m_logger);
  m_token_expander->add_token_type( new kwiver::vital::token_type_env() );
  m_token_expander->add_token_type( new kwiver::vital::token_type_sysenv() );
  m_token_expander->add_token_type( m_symtab );
  m_token_expander->add_token_type( new kwiver::vital::token_type_config( m_ref_config ) );
}


bakery_base
::~bakery_base()
{
}


// ------------------------------------------------------------------
void
bakery_base
::operator () (config_pipe_block const& config_block)
{
  kwiver::vital::config_block_key_t const root_key = flatten_keys(config_block.key);

  config_values_t const& values = config_block.values;

  for (config_value_t const& value : values)
  {
    register_config_value(root_key, value);
  }
}


// ------------------------------------------------------------------
void
bakery_base
::operator () (process_pipe_block const& process_block)
{
  config_values_t const& values = process_block.config_values;

  for (config_value_t const& value : values)
  {
    register_config_value(process_block.name, value);
  }

  m_processes.push_back(process_decl_t(process_block.name, process_block.type));
}


// ------------------------------------------------------------------
void
bakery_base
::operator () (connect_pipe_block const& connect_block)
{
  m_connections.push_back(process::connection_t(connect_block.from, connect_block.to));
}


// ------------------------------------------------------------------
/**
 * @brief Create internal config representation.
 *
 * This method creates in internal representation of a config
 * entry. The key portion starts with the supplied root_key and the
 * individual entry key is appended.
 *
 * @param root_key Key from "config" entry
 * @param value Internal pipe block
 */
void
bakery_base
::register_config_value(kwiver::vital::config_block_key_t const& root_key,
                        config_value_t const& value)
{
  kwiver::vital::config_block_key_t const subkey = flatten_keys(value.key_path);
  kwiver::vital::config_block_key_t const full_key = root_key + kwiver::vital::config_block::block_sep + subkey;
  bool is_readonly = false;
  bool is_relativepath = false;
  bool is_local_assign = false;

  // If there are options, process each one
  if ( ! value.flags.empty() )
  {
    for (config_flag_t const& flag_v : value.flags)
    {
      // normalize the case of attributes for comparison.
      std::string flag = kwiversys::SystemTools::LowerCase( flag_v );
      if (flag == flag_read_only)
      {
        is_readonly = true;
      }
      else if (flag == flag_relativepath)
      {
        is_relativepath = true;
      }
      else if (flag == flag_local_assign )
      {
        // Add key,value to local symbol table
        //
        // Note that we do not add full key. The := operator creates a
        // local symbol definition that is not related to the
        // surrounding config context.
        //
        m_symtab->add_entry( subkey, value.value );
        is_local_assign = true;
      }
      else if (flag == flag_tunable)
      {
        // Ignore here (but don't error).
      }
      else
      {
        VITAL_THROW( unrecognized_config_flag_exception, full_key, flag);
      }
    } // end foreach over flags
  }

  // If this is not a local assignment, then expand tokens
  std::string config_value = value.value;
  if ( ! is_local_assign )
  {
    try
    {
      config_value = m_token_expander->expand_token( config_value );
    }
    catch ( const provider_error_exception &e )
    {
      // Rethrow exception after adding location
      VITAL_THROW( provider_error_exception, e.what(), value.loc );
    }
  }

  // Add this entry to the ref_config so it is available for the
  // token_expander.  These config entries must be processed in the
  // order they were read from the file rather than sorted key order
  // because they can only do backward references for the config keys
  // based on file order.
  //
  // If the requested config fill-in is not in the ref_config, then it
  // must be an invalid forward reference.
  m_ref_config->set_value( full_key, config_value );

  config_info_t const info = config_info_t(config_value,
                                           is_readonly,
                                           is_relativepath,
                                           value.loc );

  config_decl_t const decl = config_decl_t(full_key, info);

  m_configs.push_back(decl);
}


// ------------------------------------------------------------------
bakery_base::config_info_t
::config_info_t(const kwiver::vital::config_block_value_t& val,
                bool                                  ro,
                bool                                  rel_path,
                const kwiver::vital::source_location& loc)
  : value(val)
  , read_only(ro)
  , relative_path(rel_path)
  , defined_loc(loc)
{
}


bakery_base::config_info_t
::~config_info_t()
{
}

// ==================================================================
// Static methods
// ------------------------------------------------------------------
kwiver::vital::config_block_key_t
bakery_base::
flatten_keys(kwiver::vital::config_block_keys_t const& keys)
{
  return kwiver::vital::join(keys, kwiver::vital::config_block::block_sep);
}


// ------------------------------------------------------------------
/**
 * @brief Convert internal config_info to real config entry
 *
 * This method converts the raw config entry taken from the internal
 * pipe representation and converts it to a config block.
 *
 * @param configs List of config_info objects to convert
 *
 * @return A config block containing the entries from the input.
 */
kwiver::vital::config_block_sptr
bakery_base::
extract_configuration_from_decls( bakery_base::config_decls_t& configs )
{
  kwiver::vital::config_block_sptr conf = kwiver::vital::config_block::empty_config();

  for( bakery_base::config_decl_t& decl : configs )
  {
    kwiver::vital::config_block_key_t const& key = decl.first;
    bakery_base::config_info_t const& info = decl.second;
    kwiver::vital::config_block_value_t val = info.value;

    if ( info.relative_path)
    {
      if ( info.defined_loc.valid() )
      {
        // Prepend CWD to val
        const std::string cwd = kwiversys::SystemTools::GetFilenamePath( info.defined_loc.file() );
        val = cwd + "/" + val;

        conf->set_location( key, info.defined_loc );
      }
      else
      {
        VITAL_THROW( relativepath_exception,
                     "Can not resolve relative path because original source file is not known.",
                     info.defined_loc );
      }
    }

    // create config entry
    conf->set_value( key, val );

    // Set location if available
    if ( info.defined_loc.valid() )
    {
      conf->set_location( key, info.defined_loc );
    }

    if ( info.read_only )
    {
      conf->mark_read_only( key );
    }
  } // end foreach

  return conf;
} // extract_configuration_from_decls

} // end namespace
