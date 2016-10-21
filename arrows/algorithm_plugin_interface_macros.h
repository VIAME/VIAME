/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
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
 * \file
 * \brief Algorithm definition type registration helper macros
 */

#ifndef ALGORITHMS_PLUGIN_INTERFACE_ALGORITHM_PLUGIN_INTERFACE_MACROS_H_
#define ALGORITHMS_PLUGIN_INTERFACE_ALGORITHM_PLUGIN_INTERFACE_MACROS_H_

#include <iostream>
#include <vital/registrar.h>
#include <vital/logger/logger.h>

// Helper macros for algorithm registration
/// Initialize required variable for algorithm type registration
/**
 * Side effect: Defines the integer variables ``algorithms_api_expected_``,
 * ``algorithms_api_registered_`` and ``algorithms_api_registrar_``. Its probably not a good
 * idea to use these variable names in the current scope, unless expecting to
 * reference the ones defined here.
 *
 * \param reg The registrar we will be registering with.
 */
#define REGISTRATION_INIT( reg ) \
  unsigned int algorithms_api_expected_ = 0, algorithms_api_registered_ = 0; \
  kwiver::vital::registrar &algorithms_api_registrar_ = reg


/// Log a summary of registration results
#define REGISTRATION_SUMMARY()                                          \
  LOG_DEBUG( kwiver::vital::get_logger( "algorithms::algorithm_plugin_interface_macros" ), \
    "REGISTRATION_SUMMARY] Registered " << algorithms_api_registered_   \
    << " of " << algorithms_api_expected_ << " algorithms\n" \
    << "\t(@" << __FILE__ << ")" );

/// Return the number of registrations that failed (int).
#define REGISTRATION_FAILURES()   (algorithms_api_expected_ - algorithms_api_registered_)


/**
 * \brief Given a algorithms::algorithm_def type, attempt registration with the
 *        given registrar
 * \param type Algorithm definition type
 */
#define REGISTER_TYPE( type ) \
  ++algorithms_api_expected_; \
  algorithms_api_registered_ += type::register_self( algorithms_api_registrar_ )


#endif // ALGORITHMS_PLUGIN_INTERFACE_ALGORITHM_PLUGIN_INTERFACE_MACROS_H_
