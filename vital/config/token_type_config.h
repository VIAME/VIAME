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

#ifndef _TOKEN_TYPE_CONFIG_H_
#define _TOKEN_TYPE_CONFIG_H_

#include <vital/util/token_type.h>

#include <vital/config/vital_config_export.h>
#include <vital/config/config_block.h>


namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/** Config token type.
 *
 * This class implements token_expander access to a config block. The
 * name of the config entry is replaced with its contents.
 *
 * The config entry passed to the constructor is still under the
 * control of the originator and will not be deleted by this class.
 *
 * When the string "$CONFIG{key}" is found in the input text is is
 * replaces with the value in the config block specified by the key.
 *
 * Example:
\code
kwiver::vital::config_block block;
kwiver::vital::token_expander m_token_expander;

m_token_expander.add_token_type( new kwiver::vital::token_type_config( block ) );
\endcode
 */
class VITAL_CONFIG_EXPORT token_type_config
  : public token_type
{
public:
  /** Constructor. A token type object is created that has access to
   * the supplied config block. The ownership of this config block
   * remains with the creator.
   *
   * @param[in] blk - config block
   */
  token_type_config( kwiver::vital::config_block_sptr blk );
  virtual ~token_type_config();

  /** Lookup name in token type resolver.
   */
  virtual bool lookup_entry (std::string const& name, std::string& result) const;


private:
  kwiver::vital::config_block_sptr m_config;

}; // end class token_type_config

} } // end namespace

#endif /* _TOKEN_TYPE_CONFIG_H_ */
