/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "config.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/none.hpp>

#include <algorithm>
#include <sstream>

namespace vistk
{

char const config::block_sep = ':';
config::key_t const config::global_value = key_t("_global");

static bool does_not_begin_with(config::key_t const& key, config::key_t const& name);

config_t
config
::empty_config(key_t const& name)
{
  return config_t(new config(name));
}

config
::~config()
{
}

config_t
config
::subblock(key_t const& key) const
{
  config_t config = empty_config(key);
  key_t const block_start = key + block_sep;
  key_t const global_start = global_value + block_sep;

  store_t::const_iterator i = m_store.begin();
  store_t::const_iterator i_end = m_store.end();

  for (; i != i_end; ++i)
  {
    if (boost::starts_with(i->first, block_start))
    {
      config->set_value(i->first.substr(block_start.size()), i->second);
    }
    else if (boost::starts_with(i->first, global_start))
    {
      config->set_value(i->first.substr(global_start.size()), i->second);
    }
  }

  return config;
}

config_t
config
::subblock_view(key_t const& key)
{
  return config_t(new config(key, this));
}

void
config
::set_value(key_t const& key, value_t const& value)
{
  if (m_parent)
  {
    m_parent->set_value(m_name + block_sep + key, value);
  }
  else
  {
    m_store[key] = value;
  }
}

void
config
::unset_value(key_t const& key)
{
  if (m_parent)
  {
    m_parent->unset_value(m_name + block_sep + key);
  }
  else
  {
    m_store.erase(key);
  }
}

config::keys_t
config
::available_values() const
{
  keys_t keys;

  if (m_parent)
  {
    keys = m_parent->available_values();

    keys_t::iterator const i = std::remove_if(keys.begin(), keys.end(), boost::bind(does_not_begin_with, _1, m_name));

    keys.erase(i, keys.end());
  }
  else
  {
    BOOST_FOREACH (store_t::value_type const& value, m_store)
    {
      keys.push_back(value.first);
    }
  }

  return keys;
}

bool
config
::has_value(key_t const& key) const
{
  if (m_parent)
  {
    return m_parent->has_value(m_name + block_sep + key);
  }

  return (m_store.find(key) != m_store.end());
}

config
::config(key_t const& name, config* parent)
  : m_parent(parent)
  , m_name(name)
  , m_store()
{
}

boost::optional<config::value_t>
config
::find_value(key_t const& key) const
{
  if (!has_value(key))
  {
    return boost::none;
  }

  return get_value(key);
}

config::value_t
config
::get_value(key_t const& key) const
{
  if (m_parent)
  {
    return m_parent->get_value(m_name + block_sep + key);
  }

  store_t::const_iterator i = m_store.find(key);

  if (i == m_store.end())
  {
    return value_t();
  }

  return i->second;
}

configuration_exception
::configuration_exception() throw()
  : pipeline_exception()
{
}

configuration_exception
::~configuration_exception() throw()
{
}

no_such_configuration_value
::no_such_configuration_value(config::key_t const& key) throw()
  : configuration_exception()
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "There is no configuration value with the key "
       << "\'" << m_key << "\'.";

  m_what = sstr.str();
}

no_such_configuration_value
::~no_such_configuration_value() throw()
{
}

char const*
no_such_configuration_value
::what() const throw()
{
  return m_what.c_str();
}

bad_configuration_cast
::bad_configuration_cast(config::key_t const& key, config::value_t const& value, char const* type, char const* reason) throw()
  : configuration_exception()
  , m_key(key)
  , m_value(value)
  , m_type(type)
  , m_reason(reason)
{
  std::ostringstream sstr;

  sstr << "Failed to cast key \'" << m_key << "\' "
       << "with value \'" << m_value << "\' as "
       << "a \'" << m_type << "\': " << m_reason << ".";

  m_what = sstr.str();
}

bad_configuration_cast
::~bad_configuration_cast() throw()
{
}

char const*
bad_configuration_cast
::what() const throw()
{
  return m_what.c_str();
}

bool
does_not_begin_with(config::key_t const& key, config::key_t const& name)
{
  return (!boost::starts_with(key, name + config::block_sep) &&
          !boost::starts_with(key, config::global_value + config::block_sep));
}

} // end namespace vistk
