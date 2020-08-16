/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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

#ifndef SPROKIT_PIPELINE_UTIL_BAKERY_DISPLAY_H
#define SPROKIT_PIPELINE_UTIL_BAKERY_DISPLAY_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include "cluster_bakery.h"
#include "pipe_bakery.h"

#include <vital/util/string.h>

#include <string>
#include <ostream>

namespace sprokit {

// ----------------------------------------------------------------------------
/**
 * \brief Formatter for pipeline bakery.
 *
 * This class formats
 */
class SPROKIT_PIPELINE_UTIL_EXPORT bakery_display
{
public:
  bakery_display( std::ostream& str );

    /**
   * \brief Format bakery blocks in simple text format.
   *
   * \param bakery Reference to bakery base.
   */
  void print( bakery_base const& bakery );
  void print( cluster_bakery const& bakery );

  /**
   * \brief Set line prefix for printing.
   *
   * This prefix string is pre-pended to each line printed to allow
   * for generating comment style output or any other creative
   * application. Defaults to the empty string.
   *
   * \param pfx The prefix string.
   */
  void set_prefix( std::string const& pfx );

  /**
   * \brief Set option to generate source location.
   *
   * The source location is the full file name and line number where
   * the element was defined in the pipeline file. The display of the
   * location can be fairly long and adversely affect readability, but
   * sometimes it is needed when debugging a pipeline file.
   *
   * \param opt TRUE will generate the source location, FALSE will not.
   */
  void generate_source_loc( bool opt );

private:
  std::ostream& m_ostr;
  std::string m_prefix;
  bool m_gen_source_loc;
};

} // end namespace

#endif
