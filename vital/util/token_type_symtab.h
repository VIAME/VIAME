// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _TOKEN_TYPE_SYMTAB_H_
#define _TOKEN_TYPE_SYMTAB_H_

#include "token_type.h"

#include <vital/util/vital_util_export.h>

#include <map>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/** Symbol table token expander.
 *
 * This token expander replaces one string with another.
 *
 * The defult name for this token type should be sufficient for most
 * users, but clever naming can have one of these symbol tables
 * masquerade as another fixed name token type, such as "ENV".
 *
 * For example, if you want to force a specific value into a file that
 * was initially expanded over the environment, a symtab can be
 * created that will do that.
 */
class VITAL_UTIL_EXPORT token_type_symtab
  : public token_type
{
public:
  token_type_symtab(std::string const& name = "SYMTAB");
  virtual ~token_type_symtab();

  /** Lookup name in token type resolver.
   */
  virtual bool lookup_entry (std::string const& name, std::string& result) const;

  /** Add entry to table.
   */
  virtual void add_entry (std::string const& name, std::string const& value);

  virtual void remove_entry (std::string const& name);

private:
  std::map < std::string, std::string > m_table;

}; // end class token_type_symtab

} } // end namespace

#endif /* _TOKEN_TYPE_SYMTAB_H_ */
