// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _TOKEN_TYPE_H_
#define _TOKEN_TYPE_H_

#include <vital/util/vital_util_export.h>
#include <string>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/** Abstract base class for token types.
 *
 *
 */
class VITAL_UTIL_EXPORT token_type
{
public:
  virtual ~token_type();

  /** Return our token type name. This is used to retrieve the name of
   * this token type when it is added to the token expander.
   */
  std::string const& token_type_name() const;

  /** Lookup name in token type resolver.
   * @param[in] name Name to look up
   * @param[out] result Translated string
   * @return TRUE if name found in table; false otherwise
   */
  virtual bool lookup_entry (std::string const& name, std::string& result) const = 0;

protected:
  token_type(std::string const& name);

private:
  std::string m_typeName;

}; // end class token_type

} } // end namespace

#endif /* _TOKEN_TYPE_H_ */
