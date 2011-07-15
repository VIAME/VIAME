/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_CONFIG_H
#define VISTK_PIPELINE_CONFIG_H

#include "pipeline-config.h"

#include "types.h"

#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <boost/utility.hpp>

#include <map>
#include <string>
#include <typeinfo>

namespace vistk
{

/**
 * \class config
 *
 * \brief Stores configuration values for use within a \ref pipeline.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT config
  : boost::noncopyable
{
  public:
    /// The type that represents a configuration value key.
    typedef std::string key_t;
    /// The type that represents a stored configuration value.
    typedef std::string value_t;

    /**
     * \brief Creates an empty configuration.
     *
     * \param name The name of the configuration block.
     */
    static config_t empty_config(key_t const& name = "");

    /**
     * \brief Destructor.
     */
    ~config();

    /**
     * \brief Get a subblock from the configuration.
     *
     * Retrieve an unlinked configuration subblock from the current
     * configuration. Changes made to it do not affect \c *this.
     *
     * \param key The name of the sub-configuration to retrieve.
     */
    config_t subblock(key_t const& key) const;

    /**
     * \brief Get a subblock view into the configuration.
     *
     * Retrieve a view into the current configuration. Changes made to \c *this
     * are seen through the view and vice versa.
     *
     * \param key The name of the sub-configuration to retrieve.
     */
    config_t subblock_view(key_t const& key);

    /**
     * \brief Internally casts the value.
     *
     * \throw no_such_configuration_value Thrown if the requested index does not exist.
     * \throw bad_configuration_cast Thrown if the cast fails.
     *
     * \param key The index of the configuration value to retrieve.
     */
    template <typename T>
    T get_value(key_t const& key) const;

    /**
     * \brief Casts the value, returning a default value in case of an error.
     *
     * \param key The index of the configuration value to retrieve.
     * \param def The value \p key does not exist or the cast fails.
     */
    template <typename T>
    T get_value(key_t const& key, T const& def) const;

    /**
     * \brief Sets a value within the configuration.
     *
     * \param key The index of the configuration value to set.
     * \param value The value to set for the \p key.
     */
    void set_value(key_t const& key, value_t const& value);

    /**
     * \brief Removes a value from the configuration.
     *
     * \param key The index of the configuration value to unset.
     */
    void unset_value(key_t const& key);

    /**
     * \brief Returns true if the value is set in the configuration.
     *
     * \param key The index of the configuration value to check.
     */
    bool has_value(key_t const& key) const;
  private:
    config(key_t const& name, config* parent = NULL);

    boost::optional<value_t> find_value(key_t const& key) const;
    value_t get_value(key_t const& key) const;

    typedef std::map<key_t, value_t> store_t;

    config* m_parent;
    key_t m_name;
    store_t m_store;

    static char const BLOCK_SEP;
    static key_t const GLOBAL_VALUE;
};

/**
 * \class configuration_exception
 *
 * \brief The base class for all exceptions thrown from \ref config.
 */
class VISTK_PIPELINE_EXPORT configuration_exception
  : public pipeline_exception
{
  public:
    configuration_exception() throw();
    ~configuration_exception() throw();
};

/**
 * \class no_such_configuration_value
 *
 * \brief Thrown when a value is requested for a value which does not exist.
 */
class VISTK_PIPELINE_EXPORT no_such_configuration_value
  : public configuration_exception
{
  public:
    no_such_configuration_value(config::key_t const& key) throw();
    ~no_such_configuration_value() throw();

    /// The requested key name.
    config::key_t const m_key;

    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class bad_configuration_cast
 *
 * \brief Thrown when a value cannot be converted to the requested type.
 */
class VISTK_PIPELINE_EXPORT bad_configuration_cast
  : public configuration_exception
{
  public:
    bad_configuration_cast(config::key_t const& key, config::value_t const& value, char const* type, char const* reason) throw();
    ~bad_configuration_cast() throw();

    /// The requested key name.
    config::key_t const m_key;
    /// The value of the requested key.
    config::value_t const m_value;
    /// The type requested for the cast.
    std::string const m_type;
    /// The reason for the failed cast.
    std::string const m_reason;

    char const* what() const throw();
  private:
    std::string m_what;
};

template <typename T>
T
config
::get_value(key_t const& key) const
{
  boost::optional<value_t> value = find_value(key);

  if (!value)
  {
    throw no_such_configuration_value(key);
  }

  try
  {
    return boost::lexical_cast<T>(*value);
  }
  catch (boost::bad_lexical_cast& e)
  {
    throw bad_configuration_cast(key, *value, typeid(T).name(), e.what());
  }
}

template <typename T>
T
config
::get_value(key_t const& key, T const& def) const
{
  try
  {
    return get_value<T>(key);
  }
  catch (...)
  {
    return def;
  }
}

} // end namespace vistk

#endif // VISTK_PIPELINE_CONFIG_H
