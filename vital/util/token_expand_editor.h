// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_UTIL_TOKEN_EXPAND_EDITOR_H
#define VITAL_UTIL_TOKEN_EXPAND_EDITOR_H

#include <vital/util/string_editor.h>

#include <vital/util/token_expander.h>

namespace kwiver {
namespace vital {

namespace edit_operation {

// ----------------------------------------------------------------
/**
 * @brief String editor that does token/macro expansion.
 *
 */
class VITAL_UTIL_EXPORT token_expand_editor
{
public:
  // -- CONSTRUCTORS --
  token_expand_editor();
  virtual ~token_expand_editor();
  virtual bool process( std::string& line );

  /**
   * @brief Add additional token type expander.
   *
   * Add an additional token expander to the collection. This editor
   * takes ownership of the specified object and will delete it when
   * being destroyed.
   *
   * @param tt New expander object.
   */
  void add_expander( kwiver::vital::token_type * tt );

private:
    token_expander m_token_expander;

}; // end class token_expand_editor

} } } // end namespace

#endif // VITAL_UTIL_TOKEN_EXPAND_EDITOR_H
