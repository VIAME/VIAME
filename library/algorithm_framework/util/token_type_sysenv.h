// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _TOKEN_TYPE_SYSENV_H_
#define _TOKEN_TYPE_SYSENV_H_

#include "token_type.h"

#include <viame/algorithm_framework/util/vital_util_export.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// What `$SYSENV{...}` resolves to in a config or pipeline file.
///
/// Eighteen names: the working directory, the host and its domain, the
/// operating system's name, version, platform and family, the processor
/// count, four memory figures in megabytes, the home directory and the
/// process id.
///
/// This used to hold a `kwiversys::SystemInformation`, which is why the
/// three checks it needed ran in the constructor. P8-T05 asks the operating
/// system directly, at the moment the question is asked, so there is nothing
/// to hold and nothing to run up front.
class VITAL_UTIL_EXPORT token_type_sysenv
  : public token_type
{
public:
  token_type_sysenv();
  virtual ~token_type_sysenv();

  /// Lookup name in token type resolver.
  virtual bool lookup_entry(
    std::string const& name,
    std::string& result ) const;
}; // end class token_type_sysenv

} // end namespace

} // end namespace

#endif // _TOKEN_TYPE_SYSENV_H_
