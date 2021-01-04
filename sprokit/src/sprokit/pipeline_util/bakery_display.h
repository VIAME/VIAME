// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
