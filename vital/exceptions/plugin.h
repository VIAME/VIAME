/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief Interface for plugin exceptions
 */

#ifndef VITAL_CORE_EXCEPTION_PLUGIN_H
#define VITAL_CORE_EXCEPTION_PLUGIN_H

#include <vital/exceptions/base.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Generic plugin exception
class VITAL_EXPORT plugin_exception
  : public vital_core_base_exception
{
public:
  /// Constructor
  plugin_exception() VITAL_NOTHROW;

  /// Destructor
  virtual ~plugin_exception() VITAL_NOTHROW;
};


// ------------------------------------------------------------------
/// Requested factory not found.
class VITAL_EXPORT plugin_factory_not_found
  : public plugin_exception
{
public:
  /// Constructor
  plugin_factory_not_found( std::string const& msg ) VITAL_NOTHROW;

  /// Destructor
  virtual ~plugin_factory_not_found() VITAL_NOTHROW;
};


// ------------------------------------------------------------------
/// Unable to create desired type.
class VITAL_EXPORT plugin_factory_type_creation_error
  : public plugin_exception
{
public:
  /// Constructor
  plugin_factory_type_creation_error( std::string const& msg ) VITAL_NOTHROW;

  /// Destructor
  virtual ~plugin_factory_type_creation_error() VITAL_NOTHROW;
};


// ------------------------------------------------------------------
/// Plugin already registered
class VITAL_EXPORT plugin_already_exists
  : public plugin_exception
{
public:
  /// Constructor
  plugin_already_exists( std::string const& msg ) VITAL_NOTHROW;

  /// Destructor
  virtual ~plugin_already_exists() VITAL_NOTHROW;
};

} } // end namespace

#endif /* VITAL_CORE_EXCEPTION_PLUGIN_H */
