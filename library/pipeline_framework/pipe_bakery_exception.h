// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file pipe_bakery_exception.h
 *
 * \brief Header for exceptions used when baking a pipeline.
 */

#ifndef SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_EXCEPTION_H
#define SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_EXCEPTION_H

#include<viame/pipeline_framework/sprokit_pipeline_util_export.h>

#include "pipe_declaration_types.h"

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/pipeline_framework/types.h>

#include <string>

namespace viame {
  class source_location;
} // namespace viame

namespace viame::pipeline {

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
    unrecognized_config_flag_exception(viame::config_block_key_t const& key, config_flag_t const& flag) noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~unrecognized_config_flag_exception() noexcept;

    /// The key the flag was on.
    viame::config_block_key_t const m_key;

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
    config_flag_mismatch_exception(viame::config_block_key_t const& key, std::string const& reason) noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~config_flag_mismatch_exception() noexcept;

    /// The key the flag was on.
    viame::config_block_key_t const m_key;

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
                          const viame::source_location& loc) noexcept;

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
                            const viame::source_location& loc) noexcept;

  provider_error_exception( const std::string& msg ) noexcept;

  /**
   * \brief Destructor.
   */
  virtual ~provider_error_exception() noexcept;

};

}

#endif // SPROKIT_PIPELINE_UTIL_PIPE_BAKERY_EXCEPTION_H
