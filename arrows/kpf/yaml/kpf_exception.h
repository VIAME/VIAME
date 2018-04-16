/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * \brief VITAL Exceptions pertaining to KPF parsing
 */

#ifndef INCL_KPF_EXCEPTIONS_H
#define INCL_KPF_EXCEPTIONS_H

#include <string>

#include <arrows/kpf/yaml/kpf_yaml_export.h>

#include <vital/exceptions/base.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Generic kpf exception
class  KPF_YAML_EXPORT kpf_exception
  : public vital_exception
{
public:
  /// Constructor
  kpf_exception() noexcept;
  /// Destructor
  virtual ~kpf_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for not enough tokens to complete parse
/**
 * Example: attempting to parse a geometry string (needs four tokens)
 * but only two tokens are left.
 */
class  KPF_YAML_EXPORT kpf_token_underrun_exception
  : public kpf_exception
{
public:
  /// Constructor
  /**
   * \param message     Description of the parsing circumstances
   */
  kpf_token_underrun_exception(std::string const& message) noexcept;
  /// Destructor
  virtual ~kpf_token_underrun_exception() noexcept;

  /// Given error message string
  std::string m_message;
};

} // ...vital
} // ...kwiver
#endif
