// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
