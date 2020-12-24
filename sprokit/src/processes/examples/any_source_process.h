// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_ANY_SOURCE_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_ANY_SOURCE_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file any_source_process.h
 *
 * \brief Declaration of the any source process.
 */

namespace sprokit {

/**
 * \class any_source_process
 *
 * \brief Generates arbitrary data.
 *
 * \process Generates arbitrary data.
 *
 * \oports
 *
 * \oport{data} The data.
 *
 * \reqs
 *
 * \req The \port{data} output must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT any_source_process
  : public process
{
public:
  PLUGIN_INFO( "any_source",
               "A process which creates arbitrary data" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  any_source_process(kwiver::vital::config_block_sptr const &config);
  /**
   * \brief Destructor.
   */
  ~any_source_process();

protected:
  /**
   * \brief Step the process.
   */
  void _step() override;

private:
  class priv;
  std::unique_ptr<priv> d;
};

} // namespace sprokit

#endif // SPROKIT_PROCESSES_EXAMPLES_ANY_SOURCE_PROCESS_H
