/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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
