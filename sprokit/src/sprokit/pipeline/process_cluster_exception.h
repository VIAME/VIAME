/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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

#ifndef SPROKIT_PIPELINE_PROCESS_CLUSTER_EXCEPTION_H
#define SPROKIT_PIPELINE_PROCESS_CLUSTER_EXCEPTION_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "process_exception.h"
#include "types.h"

#include <string>

/**
 * \file process_cluster_exception.h
 *
 * \brief Header for exceptions used within \link sprokit::process_cluster process clusters\endlink.
 */

namespace sprokit
{

/**
 * \class process_cluster_exception process_cluster_exception.h <sprokit/pipeline/process_cluster_exception.h>
 *
 * \brief The base class for special exceptions thrown from a \ref process_cluster.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT process_cluster_exception
  : public process_exception
{
  public:
    /**
     * \brief Constructor.
     */
    process_cluster_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~process_cluster_exception() noexcept;
};

/**
 * \class mapping_after_process_exception process_cluster_exception.h <sprokit/pipeline/process_cluster_exception.h>
 *
 * \brief Thrown when a configuration is mapped after the process has been created.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT mapping_after_process_exception
  : public process_cluster_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param key The name of the configuration.
     * \param mapped_name The name of the process.
     * \param mapped_key The name of the configuration.
     */
    mapping_after_process_exception(process::name_t const& name,
                                    kwiver::vital::config_block_key_t const& key,
                                    process::name_t const& mapped_name,
                                    kwiver::vital::config_block_key_t const& mapped_key) noexcept;
    /**
     * \brief Destructor.
     */
    ~mapping_after_process_exception() noexcept;

    /// The name of the \ref process_cluster the mapping occurred in.
    process::name_t const m_name;
    /// The key of the configuration on the cluster.
    kwiver::vital::config_block_key_t const m_key;
    /// The name of the \ref process which was being mapped to.
    process::name_t const m_mapped_name;
    /// The key of the configuration on the \ref process being mapped to.
    kwiver::vital::config_block_key_t const m_mapped_key;
};

/**
 * \class mapping_to_read_only_value_exception process_cluster_exception.h <sprokit/pipeline/process_cluster_exception.h>
 *
 * \brief Thrown when a configuration is mapped to a read-only value.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_EXPORT mapping_to_read_only_value_exception
  : public process_cluster_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param name The name of the process.
     * \param key The name of the configuration.
     * \param mapped_name The name of the process.
     * \param mapped_key The name of the configuration.
     */
    mapping_to_read_only_value_exception(process::name_t const& name,
                                         kwiver::vital::config_block_key_t const& key,
                                         kwiver::vital::config_block_value_t const& value,
                                         process::name_t const& mapped_name,
                                         kwiver::vital::config_block_key_t const& mapped_key,
                                         kwiver::vital::config_block_value_t const& ro_value) noexcept;
    /**
     * \brief Destructor.
     */
    ~mapping_to_read_only_value_exception() noexcept;

    /// The name of the \ref process_cluster the mapping occurred in.
    process::name_t const m_name;
    /// The key of the configuration on the cluster.
    kwiver::vital::config_block_key_t const m_key;
    /// The value of the configuration on the cluster.
    kwiver::vital::config_block_value_t const m_value;
    /// The name of the \ref process which was being mapped to.
    process::name_t const m_mapped_name;
    /// The key of the configuration on the \ref process being mapped to.
    kwiver::vital::config_block_key_t const m_mapped_key;
    /// The value of the configuration given.
    kwiver::vital::config_block_value_t const m_ro_value;
};

}

#endif // SPROKIT_PIPELINE_PROCESS_CLUSTER_EXCEPTION_H
