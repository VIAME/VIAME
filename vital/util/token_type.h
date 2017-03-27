/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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
