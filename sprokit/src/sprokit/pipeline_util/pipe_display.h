// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_UTIL_EXPORT_PIPE_H
#define SPROKIT_PIPELINE_UTIL_EXPORT_PIPE_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include <sprokit/pipeline/types.h>
#include <sprokit/pipeline_util/pipeline_builder.h>

#include <iostream>

namespace sprokit {

// ==================================================================
/**
 * @brief Export built pipeline
 *
 * This class converts a built pipeline in a readable manner as a
 * pipeline file.
 *
 * Derived classes can implement other output formats.
 */
class SPROKIT_PIPELINE_UTIL_EXPORT pipe_display
{
public:
  // -- CONSTRUCTORS --
  /**
   * @brief Create new object
   *
   * @param pipe constructed pipeline from pipeline builder.
   */
  pipe_display( std::ostream& str );
  virtual ~pipe_display();

  // display internal config blocks
  void display_pipe_blocks( const sprokit::pipe_blocks blocks );

  void print_loc( bool opt = true );

private:
  bool m_opt_print_loc{ false };
  std::ostream& m_ostr;

}; // end class pipe_display

} // end namespace

#endif // SPROKIT_PIPELINE_UTIL_PIPE_DISPLAY_H
