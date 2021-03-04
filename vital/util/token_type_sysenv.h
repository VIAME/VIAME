// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _TOKEN_TYPE_SYSENV_H_
#define _TOKEN_TYPE_SYSENV_H_

#include "token_type.h"

#include <vital/util/vital_util_export.h>

#include <kwiversys/SystemInformation.hxx>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/** System attributes resolver.
 *
 *
 */
class VITAL_UTIL_EXPORT token_type_sysenv
  : public token_type
{
public:
  token_type_sysenv();
  virtual ~token_type_sysenv();

  /** Lookup name in token type resolver.
   */
  virtual bool lookup_entry (std::string const& name, std::string& result) const;

private:
  kwiversys::SystemInformation m_sysinfo;

}; // end class token_type_sysenv

} // end namespace
} // end namespace

#endif /* _TOKEN_TYPE_SYSENV_H_ */
