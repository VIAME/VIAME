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

/**
 * \file pipe_bakery_exception.h
 *
 * \brief Header for exceptions used when baking a pipeline.
 */

#ifndef SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_EXCEPTION_H
#define SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_EXCEPTION_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include "pipe_declaration_types.h"

#include <vital/config/config_block.h>
#include <sprokit/pipeline/types.h>

#include <string>

namespace kwiver {
namespace vital {

  class source_location;

} // end namespace vital
} // end namespace kwiver


namespace sprokit {

// ------------------------------------------------------------------
/**
 * \class pipe_bakery_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The base class for all exceptions thrown when baking a pipeline.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT pipe_bakery_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    pipe_bakery_exception() noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~pipe_bakery_exception() noexcept;
};

// ------------------------------------------------------------------
/**
 * \class missing_cluster_block_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when baking a cluster without a cluster block.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT missing_cluster_block_exception
  : public pipe_bakery_exception
{
  public:
    /**
     * \brief Constructor.
     */
    missing_cluster_block_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~missing_cluster_block_exception() noexcept;
};

// ------------------------------------------------------------------
/**
 * \class multiple_cluster_blocks_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when baking a cluster with multiple cluster blocks.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT multiple_cluster_blocks_exception
  : public pipe_bakery_exception
{
  public:
    /**
     * \brief Constructor.
     */
    multiple_cluster_blocks_exception() noexcept;
    /**
     * \brief Destructor.
     */
    ~multiple_cluster_blocks_exception() noexcept;
};

// ------------------------------------------------------------------
/**
 * \class cluster_without_processes_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when a cluster does not contain any processes.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT cluster_without_processes_exception
  : public pipe_bakery_exception
{
  public:
    /**
     * \brief Constructor.
     */
    cluster_without_processes_exception() noexcept;

    /**
     * \brief Destructor.
     */
    virtual ~cluster_without_processes_exception() noexcept;
};

// ------------------------------------------------------------------
/**
 * \class cluster_without_ports_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when a cluster does not contain any ports.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT cluster_without_ports_exception
  : public pipe_bakery_exception
{
  public:
    /**
     * \brief Constructor.
     */
    cluster_without_ports_exception() noexcept;

    /**
     * \brief Destructor.
     */
    virtual ~cluster_without_ports_exception() noexcept;
};

// ------------------------------------------------------------------
/**
 * \class duplicate_cluster_port_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when a cluster is declared with duplicate ports.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT duplicate_cluster_port_exception
  : public pipe_bakery_exception
{
  public:
    /**
     * \brief Constructor.
     */
    duplicate_cluster_port_exception(process::port_t const& port, char const* const side) noexcept;

    /**
     * \brief Destructor.
     */
    virtual ~duplicate_cluster_port_exception() noexcept;

    /// The name of the duplicate port.
    process::port_t const m_port;
};

// ------------------------------------------------------------------
/**
 * \class duplicate_cluster_input_port_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when a cluster is declared with duplicate input ports.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT duplicate_cluster_input_port_exception
  : public duplicate_cluster_port_exception
{
  public:
    /**
     * \brief Constructor.
     */
    duplicate_cluster_input_port_exception(process::port_t const& port) noexcept;

    /**
     * \brief Destructor.
     */
    virtual ~duplicate_cluster_input_port_exception() noexcept;
};

// ------------------------------------------------------------------
/**
 * \class duplicate_cluster_output_port_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when a cluster is declared with duplicate output ports.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT duplicate_cluster_output_port_exception
  : public duplicate_cluster_port_exception
{
  public:
    /**
     * \brief Constructor.
     */
    duplicate_cluster_output_port_exception(process::port_t const& port) noexcept;

    /**
     * \brief Destructor.
     */
    virtual ~duplicate_cluster_output_port_exception() noexcept;
};

// ------------------------------------------------------------------
/**
 * \class unrecognized_config_flag_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when a flag on a configuration is not recognized.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT unrecognized_config_flag_exception
  : public pipe_bakery_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param key The key the flag was on.
     * \param flag The unrecognized flag.
     */
    unrecognized_config_flag_exception(kwiver::vital::config_block_key_t const& key, config_flag_t const& flag) noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~unrecognized_config_flag_exception() noexcept;

    /// The key the flag was on.
    kwiver::vital::config_block_key_t const m_key;

    /// The unrecognized flag.
    config_flag_t const m_flag;
};

// ------------------------------------------------------------------
/**
 * \class config_flag_mismatch_exception pipe_bakery_exception.h <sprokit/pipeline_util/pipe_bakery_exception.h>
 *
 * \brief The exception thrown when flags on a configuration are mismatched.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT config_flag_mismatch_exception
  : public pipe_bakery_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param key The key the flag was on.
     * \param reason The reason for the mismatch.
     */
    config_flag_mismatch_exception(kwiver::vital::config_block_key_t const& key, std::string const& reason) noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~config_flag_mismatch_exception() noexcept;

    /// The key the flag was on.
    kwiver::vital::config_block_key_t const m_key;

    /// The reason for the mismatch.
    std::string const m_reason;
};

// ------------------------------------------------------------------
class SPROKIT_PIPELINE_UTIL_EXPORT relativepath_exception
  : public pipe_bakery_exception
{
public:
  /**
   * \brief Constructor.
   */
  relativepath_exception( const std::string& msg,
                          const kwiver::vital::source_location& loc) noexcept;

  /**
   * \brief Destructor.
   */
  virtual ~relativepath_exception() noexcept;

};


// ------------------------------------------------------------------
class SPROKIT_PIPELINE_UTIL_EXPORT provider_error_exception
  : public pipe_bakery_exception
{
public:
  /**
   * \brief Constructor.
   */
  provider_error_exception( const std::string& msg,
                            const kwiver::vital::source_location& loc) noexcept;

  provider_error_exception( const std::string& msg ) noexcept;

  /**
   * \brief Destructor.
   */
  virtual ~provider_error_exception() noexcept;

};

}

#endif // SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_EXCEPTION_H
