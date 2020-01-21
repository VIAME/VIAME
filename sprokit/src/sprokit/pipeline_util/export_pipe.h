/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
