// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_UTIL_EXPORT_PIPE_H
#define SPROKIT_PIPELINE_UTIL_EXPORT_PIPE_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include <sprokit/pipeline/types.h>

#include <iostream>

namespace sprokit {

class pipeline_builder;

// ==================================================================
/**
 * @brief Export built pipeline
 *
 * This class converts a built pipeline in a readable manner as a
 * pipeline file.
 *
 * Derived classes can implement other output formats.
 */
class SPROKIT_PIPELINE_UTIL_EXPORT export_pipe
{
public:
  // -- CONSTRUCTORS --
  /**
   * @brief Create new object
   *
   * @param pipe constructed pipeline from pipeline builder.
   */
  export_pipe( const sprokit::pipeline_builder& builder );
  virtual ~export_pipe();

  // Generate output for pipeline
  virtual void generate( std::ostream& str );

private:
  const sprokit::pipeline_builder&  m_builder;
}; // end class export_pipe

} // end namespace

#endif // SPROKIT_PIPELINE_UTIL_EXPORT_PIPE_H
