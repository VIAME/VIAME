// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _TOKEN_TYPE_ENV_H_
#define _TOKEN_TYPE_ENV_H_

#include "token_type.h"
#include <vital/util/vital_util_export.h>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/** Virtual base class for token types.
 *
 *
 */
class VITAL_UTIL_EXPORT token_type_env
  : public token_type
{
public:
  token_type_env();
  virtual ~token_type_env();

  /** Lookup name in token type resolver.
   */
  virtual bool lookup_entry (std::string const& name, std::string& result) const;

}; // end class token_type_env

} } // end namespace

#endif /* _TOKEN_TYPE_ENV_H_ */
