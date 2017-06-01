/*ckwg +29
 * Copyright 2013-2016 by Kitware, Inc.
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


#ifndef _TOKEN_EXPANDER_H_
#define _TOKEN_EXPANDER_H_

#include <vital/util/vital_util_export.h>
#include <vital/logger/logger.h>

#include <map>
#include <string>
#include <memory>


namespace kwiver {
namespace vital {

class token_type;


// ----------------------------------------------------------------
/** @brief Replace tokens with text.
 *
 * This class represents a token expander or a macro expander.
 *
 * The syntax of a token is "$TYPE{name}". Where \c TYPE indicates the
 * type or class of the token, and \c name is the optional entry to use.
 *
 * In the spirit of most macro expanders, if the specified type is not
 * found and the "name" is not found, then it is replaced with the
 * null string "". If the TYPE is not found, the string is left
 * intact.
 *
@code
kwiver::vital::token_expander * exp;

exp->add_token_type( new kwiver::vital::token_type_env() );
exp->add_token_type( new kwiver::vital::token_type_sysenv() );

// Create a symtab expander
kwiver::vital::token_type_symtab* sym = new kwiver::vital::token_type_symtab();
sym.add_entry( "foo", "bar" );  // add some entries to the symbol table
sym.add_entry( "home", "on the range" );
sym.add_entry( "pi", "3.14159265358979323846264338327950288419716939937510" );

exp->add_token_type( sym );  // add to expander

@endcode
 *
 * Usage example
@code
// Derived class just loads all needed token types.
class app_expander : public kwiver::vital::token_expander
{
public:
  app_expander()
  {
    // Load the usual token types
    this->add_token_type( new kwiver::vital::token_type_env() );
    this->add_token_type( new kwiver::vital::token_type_sysenv() );
  }
};
@endcode
 */
class VITAL_UTIL_EXPORT token_expander
{
public:
  token_expander();
  virtual ~token_expander();

  /** @brief Optional singleton support.
   *
   * This class can be treated as a singleton, but is it not required.
   *
   * The singleton support helps support cases where the token
   * expander is used and there is no easy way to pass the expander
   * object.
   *
   * Typical usage is:
   *
   * In Main program at startup, create the singleton and configure
   * with desired token types.
   *
@code
kwiver::vital::token_expander * exp = new kwiver::vital::token_expander::instance();
exp->add_token_type( new kwiver::vital::token_type_env() );
exp->add_token_type( new kwiver::vital::token_type_sysenv() );
@endcode
   *
   * The token expander can be easily referenced anywhere in the code
   * without having to explicitly pass a pointer to the object.
   *
@code
std::string text = get_raw_text();
text = kwiver::vital::token_expander::instance()->expand_token( text );
@endcode
   */

  /** @brief Add new token type to expander.
   *
   * The supplied token type is added to this token expander.
   *
   * Once the object has been added, ownership is transferred to the
   * collective and it will be deleted when this token expander is
   * deleted.
   *
   * @param[in] tt - token type handler to add
   *
   * @return True indicates the handler has been added.
   */
  bool add_token_type( kwiver::vital::token_type * tt );

  /** @brief Expand tokens over a string.
   *
   * All registered tokens are expanded in the input string to produce
   * the output string.
   *
   * @param initial_string - input string with fill-in markers.
   *
   * @return String with all recognized tokens replaced.
   */
  std::string expand_token(  std::string const& initial_string );

protected:
  /**
   * @brief Handle missing provider entry.
   *
   * This method is called when a provider does not have the requested
   * item.
   *
   * @param provider Name of provider.
   * @param entry Name of entry.
   *
   * @return \b true inserts the original text leaving the unresolved
   * entry in the expanded string. \b false inserts empty string,
   * removing the unresolved entry from the output.
   */
  virtual bool handle_missing_entry( const std::string& provider, const std::string& entry );

  /**
   * @brief Handle missing provider.
   *
   * This method is called when the requested provider is not
   * registered.
   *
   * @param provider Name of provider.
   * @param entry Name of entry.
   *
   * @return \b true inserts the original text leaving the unresolved
   * entry in the expanded string. \b false inserts empty string,
   * removing the unresolved entry from the output.
   */
  virtual bool handle_missing_provider( const std::string& provider, const std::string& entry );

private:
  typedef std::map < std::string, std::shared_ptr< kwiver::vital::token_type > > map_t;
  typedef map_t::iterator iterator_t;

  map_t m_typeList;

  logger_handle_t m_logger;

}; // end class token_expander

} } // end namespace

#endif // _TOKEN_EXPANDER_H_
