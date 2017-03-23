/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#ifndef SPROKIT_PIPELINE_UTIL_PROVIDERS_H
#define SPROKIT_PIPELINE_UTIL_PROVIDERS_H

#include "pipeline_util-config.h"

#include <vital/config/config_block.h>

#include <memory>

/**
 * \file providers.h
 *
 * \brief Configuration providers.
 */

namespace sprokit
{

// ==================================================================
class provider;
/// Type to more easily handle providers.
typedef std::shared_ptr<provider> provider_t;

/**
 * \class provider providers.h "providers.h"
 *
 * \brief The base abstract class for providers.
 */
class SPROKIT_PIPELINE_UTIL_NO_EXPORT provider
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
     * \brief Process a request from the configuration for a value.
     *
     * \param index The value requested from the provider.
     *
     * \returns The dereferenced value for the \p index.
     */
    virtual kwiver::vital::config_block_value_t operator () (kwiver::vital::config_block_value_t const& index) const = 0;
};

/**
 * \class config_provider providers.h "providers.h"
 *
 * \brief A provider which dereferences configuration values (symlinking).
 */
class SPROKIT_PIPELINE_UTIL_NO_EXPORT config_provider
  : public provider
{
  public:
    /**
     * \brief Constructor.
     *
     * \param conf The configuration to use as a lookup.
     */
    config_provider(kwiver::vital::config_block_sptr const conf);
    /**
     * \brief Destructor.
     */
    ~config_provider();

    /**
     * \brief Process a request from the configuration for a value.
     *
     * \param index The value requested from the provider.
     *
     * \returns The dereferenced value for the \p index.
     */
    kwiver::vital::config_block_value_t operator () (kwiver::vital::config_block_value_t const& index) const;
  private:
    kwiver::vital::config_block_sptr const m_config;
};


// ==================================================================
/**
 * \class system_provider providers.h "providers.h"
 *
 * \brief A provider which queries information from the system.
 */
class SPROKIT_PIPELINE_UTIL_NO_EXPORT system_provider
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
     * \brief Process a request from the configuration for a value.
     *
     * \param index The value requested from the provider.
     *
     * \returns The dereferenced value for the \p index.
     */
    kwiver::vital::config_block_value_t operator () (kwiver::vital::config_block_value_t const& index) const;
};


// ==================================================================
/**
 * \class environment_provider providers.h "providers.h"
 *
 * \brief A provider which queries information from the environment.
 */
class SPROKIT_PIPELINE_UTIL_NO_EXPORT environment_provider
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
     * \brief Process a request from the configuration for a value.
     *
     * \param index The value requested from the provider.
     *
     * \returns The dereferenced value for the \p index.
     */
    kwiver::vital::config_block_value_t operator () (kwiver::vital::config_block_value_t const& index) const;
};

}

#endif // SPROKIT_PIPELINE_UTIL_PROVIDERS_H
