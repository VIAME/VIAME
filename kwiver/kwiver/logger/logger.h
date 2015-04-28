/*ckwg +29
 * Copyright 2013-2014 by Kitware, Inc.
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

#ifndef KWIVER_CORE_LOGGER_H_
#define KWIVER_CORE_LOGGER_H_

#include "kwiver_logger.h"

/**
 * @file This file defines the main user interface to the kwiver logger.
 */

namespace kwiver {

//@{
/**
 * @brief Get pointer to logger object.
 *
 * @param name Logger name
 *
 * @return Handle (pointer) to logger object.
 */
logger_handle_t KWIVER_LOGGER_EXPORT get_logger( const char * const name );
logger_handle_t KWIVER_LOGGER_EXPORT get_logger( std::string const& name );
//@}

/**
 * Logs a message with the ERROR level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_ERROR( logger, msg ) do {                        \
    if ( logger->is_error_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                 \
      logger->log_error( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )


/**
 * Logs a message with the WARN level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_WARN( logger, msg ) do {                        \
    if ( logger->is_warn_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                \
      logger->log_warn( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )


/**
 * Logs a message with the INFO level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_INFO( logger, msg ) do {                        \
    if ( logger->is_info_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                \
      logger->log_info( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )


/**
 * Logs a message with the DEBUG level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_DEBUG( logger, msg ) do {                        \
    if ( logger->is_debug_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                 \
      logger->log_debug( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )


/**
 * Logs a message with the TRACE level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_TRACE( logger, msg ) do {                        \
    if ( logger->is_trace_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                 \
      logger->log_trace( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )

/**
 * Performs assert and logs message if condition is false.  If
 * condition is false, log a message at the FATAL level is
 * generated. This is similar to the library assert except that the
 * message goes to the logger.
 * @param logger the logger to be used
 * @param cond the condition which should be met to pass the assertion
 * @param msg the message string to log.
 */
#define LOG_ASSERT( logger, cond, msg ) do {                   \
    if ( ! ( cond ) ) {                                        \
      std::stringstream _oss_; _oss_  << "ASSERTION FAILED: (" \
                                      << # cond ")\n"  << msg; \
      logger->log_error( _oss_.str(), KWIVER_LOGGER_SITE );     \
} while ( 0 )


// Test for debugging level being enabled
#define IS_FATAL_ENABLED( logger ) ( logger->is_fatal_enabled() )
#define IS_ERROR_ENABLED( logger ) ( logger->is_error_enabled() )
#define IS_WARN_ENABLED( logger )  ( logger->is_warn_enabled() )
#define IS_INFO_ENABLED( logger )  ( logger->is_info_enabled() )
#define IS_DEBUG_ENABLED( logger ) ( logger->is_debug_enabled() )
#define IS_TRACE_ENABLED( logger ) ( logger->is_trace_enabled() )

} // end namespace

#endif /* KWIVER_CORE_LOGGER_H_ */
