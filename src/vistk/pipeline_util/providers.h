/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_PROVIDERS_H
#define VISTK_PIPELINE_UTIL_PROVIDERS_H

#include "pipeline_util-config.h"

#include <vistk/pipeline/config.h>

/**
 * \file providers.h
 *
 * \brief Configuration providers.
 */

namespace vistk
{

class provider;
/// Type to more easily handle providers.
typedef boost::shared_ptr<provider> provider_t;

/**
 * \class provider providers.h "providers.h"
 *
 * \brief The base abstract class for providers.
 */
class VISTK_PIPELINE_UTIL_NO_EXPORT provider
{
  public:
    /**
     * \brief Constructor.
     */
    provider();
    /**
     * \brief Destructor.
     */
    virtual ~provider();

    /**
     * \brief Processes a request from the configuration for a value.
     *
     * \param index The value requested from the provider.
     *
     * \returns The dereferenced value for the \p index.
     */
    virtual config::value_t operator () (config::value_t const& index) const = 0;
};

/**
 * \class config_provider providers.h "providers.h"
 *
 * \brief A provider which dereferences configuration values (symlinking).
 */
class VISTK_PIPELINE_UTIL_NO_EXPORT config_provider
  : public provider
{
  public:
    /**
     * \brief Constructor.
     *
     * \param conf The configuration to use as a lookup.
     */
    config_provider(config_t const conf);
    /**
     * \brief Destructor.
     */
    ~config_provider();

    /**
     * \brief Processes a request from the configuration for a value.
     *
     * \param index The value requested from the provider.
     *
     * \returns The dereferenced value for the \p index.
     */
    config::value_t operator () (config::value_t const& index) const;
  private:
    config_t const m_config;
};

/**
 * \class system_provider providers.h "providers.h"
 *
 * \brief A provider which queries information from the system.
 */
class VISTK_PIPELINE_UTIL_NO_EXPORT system_provider
  : public provider
{
  public:
    /**
     * \brief Constructor.
     */
    system_provider();
    /**
     * \brief Destructor.
     */
    ~system_provider();

    /**
     * \brief Processes a request from the configuration for a value.
     *
     * \param index The value requested from the provider.
     *
     * \returns The dereferenced value for the \p index.
     */
    config::value_t operator () (config::value_t const& index) const;
};

/**
 * \class environment_provider providers.h "providers.h"
 *
 * \brief A provider which queries information from the environment.
 */
class VISTK_PIPELINE_UTIL_NO_EXPORT environment_provider
  : public provider
{
  public:
    /**
     * \brief Constructor.
     */
    environment_provider();
    /**
     * \brief Destructor.
     */
    ~environment_provider();

    /**
     * \brief Processes a request from the configuration for a value.
     *
     * \param index The value requested from the provider.
     *
     * \returns The dereferenced value for the \p index.
     */
    config::value_t operator () (config::value_t const& index) const;
};

}

#endif // VISTK_PIPELINE_UTIL_PROVIDERS_H
