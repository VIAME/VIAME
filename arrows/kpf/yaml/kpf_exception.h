// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
