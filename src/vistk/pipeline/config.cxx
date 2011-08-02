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

/**
 * \file config.cxx
 *
 * \brief Implementation of \link vistk::config configuration\endlink in the pipeline.
 */

namespace vistk
{

config::key_t const config::block_sep = key_t(":");
config::key_t const config::global_value = key_t("_global");

static bool does_not_begin_with(config::key_t const& key, config::key_t const& name);
static config::key_t strip_block_name(config::key_t const& key);

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
  config_t conf = empty_config(key);

  BOOST_FOREACH (key_t const& key_name, available_values())
  {
    if (does_not_begin_with(key_name, key))
    {
      continue;
    }

    key_t const stripped_key_name = strip_block_name(key_name);

    conf->set_value(stripped_key_name, get_value(key_name));
  }

  return conf;
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
    if (is_read_only(key))
    {
      throw set_on_read_only_value_exception(key, get_value<value_t>(key, value_t()), value);
    }

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
    if (is_read_only(key))
    {
      throw unset_on_read_only_value_exception(key, get_value<value_t>(key, value_t()));
    }

    m_store.erase(key);
  }
}

bool
config
::is_read_only(key_t const& key) const
{
  ro_list_t::iterator i = m_ro_list.find(key);

  return (i != m_ro_list.end());
}

void
config
::mark_read_only(key_t const& key)
{
  m_ro_list.insert(key);
}

void
config
::merge_config(config_t conf)
{
  BOOST_FOREACH (store_t::value_type const& value, conf->m_store)
  {
    set_value(value.first, value.second);
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

    std::for_each(keys.begin(), keys.end(), strip_block_name);
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

no_such_configuration_value_exception
::no_such_configuration_value_exception(config::key_t const& key) throw()
  : configuration_exception()
  , m_key(key)
{
  std::ostringstream sstr;

  sstr << "There is no configuration value with the key "
       << "\'" << m_key << "\'.";

  m_what = sstr.str();
}

no_such_configuration_value_exception
::~no_such_configuration_value_exception() throw()
{
}

char const*
no_such_configuration_value_exception
::what() const throw()
{
  return m_what.c_str();
}

bad_configuration_cast_exception
::bad_configuration_cast_exception(config::key_t const& key, config::value_t const& value, char const* type, char const* reason) throw()
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

bad_configuration_cast_exception
::~bad_configuration_cast_exception() throw()
{
}

char const*
bad_configuration_cast_exception
::what() const throw()
{
  return m_what.c_str();
}

set_on_read_only_value_exception
::set_on_read_only_value_exception(config::key_t const& key, config::value_t const& value, config::value_t const& new_value) throw()
  : configuration_exception()
  , m_key(key)
  , m_value(value)
  , m_new_value(new_value)
{
  std::ostringstream sstr;

  sstr << "The key \'" << m_key << "\' "
       << "was marked as read-only with the value "
       << "\'" << m_value << "\' was attempted to be "
       << "set to \'" << m_key << "\'.";

  m_what = sstr.str();
}

set_on_read_only_value_exception
::~set_on_read_only_value_exception() throw()
{
}

char const*
set_on_read_only_value_exception
::what() const throw()
{
  return m_what.c_str();
}

unset_on_read_only_value_exception
::unset_on_read_only_value_exception(config::key_t const& key, config::value_t const& value) throw()
  : configuration_exception()
  , m_key(key)
  , m_value(value)
{
  std::ostringstream sstr;

  sstr << "The key \'" << m_key << "\' "
       << "was marked as read-only with the value "
       << "\'" << m_value << "\' was attempted to be "
       << "unset.";

  m_what = sstr.str();
}

unset_on_read_only_value_exception
::~unset_on_read_only_value_exception() throw()
{
}

char const*
unset_on_read_only_value_exception
::what() const throw()
{
  return m_what.c_str();
}

bool
does_not_begin_with(config::key_t const& key, config::key_t const& name)
{
  static config::key_t const global_start = config::global_value + config::block_sep;

  return (!boost::starts_with(key, name + config::block_sep) &&
          !boost::starts_with(key, global_start));
}

config::key_t
strip_block_name(config::key_t const& key)
{
  size_t const pos = key.find(config::block_sep);

  if (pos == config::key_t::npos)
  {
    return key;
  }

  return key.substr(pos + 1);
}

}
