// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file orphan_process.h
 *
 * \brief Declaration of the orphan process.
 */

namespace sprokit
{

/**
 * \class orphan_process
 *
 * \brief A no-op process.
 *
 * \process A no-op process.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT orphan_process
  : public process
{
public:
  PLUGIN_INFO( "orphan",
               "A dummy process" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */

  orphan_process(kwiver::vital::config_block_sptr const& config);
  /**
   * \brief Destructor.
   */
  ~orphan_process();
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_ORPHAN_PROCESS_H
