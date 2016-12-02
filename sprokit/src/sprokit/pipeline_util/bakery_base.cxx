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
 * @file   bakery_base.cxx
 * @brief  Implementation for bakery_base class
 */

#include "bakery_base.h"

#include "pipe_bakery_exception.h"
#include "path.h"
#include "ensure_provided.h"
#include "provider_dereferencer.h"
#include "config_provider_sorter.h"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>


namespace sprokit {

namespace { // anonymous

// ------------------------------------------------------------------
void
set_config_value( kwiver::vital::config_block_sptr            conf,
                  bakery_base::config_info_t const&           flags,
                  kwiver::vital::config_block_key_t const&    key,
                  kwiver::vital::config_block_value_t const&  value )
{
  kwiver::vital::config_block_value_t val = value;

  kwiver::vital::config_block_value_t const cur_val = conf->get_value( key, kwiver::vital::config_block_value_t() );
  bool const has_cur_val = ! cur_val.empty();

  switch ( flags.append )
  {
  case bakery_base::config_info_t::append_string:
    val = cur_val + val;
    break;

  case bakery_base::config_info_t::append_comma:
    if ( has_cur_val )
    {
      val = cur_val + "," + val;
    }
    break;

  case bakery_base::config_info_t::append_space:
    if ( has_cur_val )
    {
      val = cur_val + " " + val;
    }
    break;

  case bakery_base::config_info_t::append_path:
  {
    path_t const base_path = path_t( has_cur_val ? cur_val : "." );
    path_t const val_path = path_t( val );
    path_t const new_path = base_path / val_path;

    val = new_path.string< kwiver::vital::config_block_value_t > ();
    break;
  }

  case bakery_base::config_info_t::append_none:
  default:
    break;
  }

  conf->set_value( key, val );

  if ( flags.read_only )
  {
    conf->mark_read_only( key );
  }
} // set_config_value

} // end namespace

config_flag_t const bakery_base::flag_read_only = config_flag_t("ro");
config_flag_t const bakery_base::flag_append = config_flag_t("append");
config_flag_t const bakery_base::flag_append_prefix = config_flag_t("append=");
config_flag_t const bakery_base::flag_append_comma = config_flag_t("comma");
config_flag_t const bakery_base::flag_append_space = config_flag_t("space");
config_flag_t const bakery_base::flag_append_path = config_flag_t("path");
config_flag_t const bakery_base::flag_tunable = config_flag_t("tunable");

config_provider_t const bakery_base::provider_config = config_provider_t("CONF");
config_provider_t const bakery_base::provider_environment = config_provider_t("ENV");
config_provider_t const bakery_base::provider_system = config_provider_t("SYS");


// ------------------------------------------------------------------
bakery_base
::bakery_base()
  : m_configs()
  , m_processes()
  , m_connections()
{
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

  BOOST_FOREACH (config_value_t const& value, values)
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

  BOOST_FOREACH (config_value_t const& value, values)
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
void
bakery_base
::register_config_value(kwiver::vital::config_block_key_t const& root_key, config_value_t const& value)
{
  config_key_t const key = value.key;

  kwiver::vital::config_block_key_t const subkey = flatten_keys(key.key_path);

  config_reference_t c_value;

  if (key.options.provider)
  {
    c_value = provider_request_t(*key.options.provider, value.value);
  }
  else
  {
    c_value = value.value;
  }

  kwiver::vital::config_block_key_t const full_key = root_key + kwiver::vital::config_block::block_sep + subkey;

  bool is_readonly = false;
  config_info_t::append_t append = config_info_t::append_none;

#define APPEND_CHECK(flag)                                             \
  do                                                                   \
  {                                                                    \
    if (append != config_info_t::append_none)                          \
    {                                                                  \
      std::string const reason = "The \'" + flag + "\' flag cannot "   \
                                 "be used with other appending flags"; \
                                                                       \
      throw config_flag_mismatch_exception(full_key, reason);          \
    }                                                                  \
  } while (false)

  if (key.options.flags)
  {
    BOOST_FOREACH (config_flag_t const& flag, *key.options.flags)
    {
      if (flag == flag_read_only)
      {
        is_readonly = true;
      }
      else if (flag == flag_append)
      {
        APPEND_CHECK(flag_append);

        append = config_info_t::append_string;
      }
      else if (boost::starts_with(flag, flag_append_prefix))
      {
        APPEND_CHECK(flag);

        config_flag_t const& kind = flag.substr(flag_append_prefix.size());

        if (kind == flag_append_comma)
        {
          append = config_info_t::append_comma;
        }
        else if (kind == flag_append_space)
        {
          append = config_info_t::append_space;
        }
        else if (kind == flag_append_path)
        {
          append = config_info_t::append_path;
        }
        else
        {
          throw unrecognized_config_flag_exception(full_key, flag);
        }
      }
      else if (flag == flag_tunable)
      {
        // Ignore here (but don't error).
      }
      else
      {
        throw unrecognized_config_flag_exception(full_key, flag);
      }
    }
  }

#undef APPEND_CHECK

  config_info_t const info = config_info_t(c_value, is_readonly, append);

  config_decl_t const decl = config_decl_t(full_key, info);

  m_configs.push_back(decl);
}


// ------------------------------------------------------------------
bakery_base::config_info_t
::config_info_t(config_reference_t const& ref,
                bool ro,
                append_t app)
  : reference(ref)
  , read_only(ro)
  , append(app)
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
  return boost::join(keys, kwiver::vital::config_block::block_sep);
}


// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
bakery_base::
extract_configuration_from_decls( bakery_base::config_decls_t& configs )
{
  dereference_static_providers( configs );

  kwiver::vital::config_block_sptr tmp_conf = kwiver::vital::config_block::empty_config();

  ensure_provided const ensure;

  {
    typedef std::set< kwiver::vital::config_block_key_t > unprovided_keys_t;

    unprovided_keys_t unprovided_keys;

    BOOST_FOREACH( bakery_base::config_decl_t & decl, configs )
    {
      kwiver::vital::config_block_key_t const& key = decl.first;

      if ( unprovided_keys.count( key ) )
      {
        continue;
      }

      bakery_base::config_info_t const& info = decl.second;
      bakery_base::config_reference_t const& ref = info.reference;

      kwiver::vital::config_block_value_t val;

      // Only add provided configurations to the configuration.
      try
      {
        val = boost::apply_visitor( ensure, ref );
      }
      catch ( unrecognized_provider_exception const /*e*/ )
      {
        unprovided_keys.insert( key );

        continue;
      }

      set_config_value( tmp_conf, info, key, val );
    }
  }

  // Dereference configuration providers.
  {
    config_provider_sorter sorter;

    BOOST_FOREACH( bakery_base::config_decl_t & decl, configs )
    {
      kwiver::vital::config_block_key_t const& key = decl.first;
      bakery_base::config_info_t const& info = decl.second;
      bakery_base::config_reference_t const& ref = info.reference;

      /// \bug Why must this be done?
      typedef boost::variant< kwiver::vital::config_block_key_t > dummy_variant;

      dummy_variant const var = key;

      boost::apply_visitor( sorter, var, ref );
    }

    kwiver::vital::config_block_keys_t const keys = sorter.sorted();

    provider_dereferencer const deref( tmp_conf );

    /// \todo This is algorithmically naive, but I'm not sure if there's a better way.
    BOOST_FOREACH( kwiver::vital::config_block_key_t const & key, keys )
    {
      BOOST_FOREACH( bakery_base::config_decl_t & decl, configs )
      {
        kwiver::vital::config_block_key_t const& cur_key = decl.first;

        if ( key != cur_key )
        {
          continue;
        }

        bakery_base::config_info_t& info = decl.second;
        bakery_base::config_reference_t& ref = info.reference;

        ref = boost::apply_visitor( deref, ref );

        kwiver::vital::config_block_value_t const val = boost::apply_visitor( ensure, ref );

        set_config_value( tmp_conf, info, key, val );
      }
    }
  }

  kwiver::vital::config_block_sptr conf = kwiver::vital::config_block::empty_config();

  BOOST_FOREACH( bakery_base::config_decl_t & decl, configs )
  {
    kwiver::vital::config_block_key_t const& key = decl.first;
    bakery_base::config_info_t const& info = decl.second;
    bakery_base::config_reference_t const& ref = info.reference;

    kwiver::vital::config_block_value_t val;

    try
    {
      val = boost::apply_visitor( ensure, ref );
    }
    catch ( unrecognized_provider_exception const& e )
    {
      throw unrecognized_provider_exception( key, e.m_provider, e.m_index );
    }

    set_config_value( conf, info, key, val );
  }

  return conf;
} // extract_configuration_from_decls


// ------------------------------------------------------------------
void
bakery_base::
dereference_static_providers( bakery_base::config_decls_t& configs )
{
  provider_dereferencer const deref;

  BOOST_FOREACH( bakery_base::config_decl_t & decl, configs )
  {
    bakery_base::config_info_t& info = decl.second;
    bakery_base::config_reference_t& ref = info.reference;

    ref = boost::apply_visitor( deref, ref );
  }
}




} // end namespace
