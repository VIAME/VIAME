// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_MUTATE_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_MUTATE_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file mutate_process.h
 *
 * \brief Declaration of the mutate process.
 */

namespace sprokit {

/**
 * \class mutate_process
 *
 * \brief A process with a mutate input port.
 *
 * \process A process with a mutate input port.
 *
 * \iports
 *
 * \iport{mutate} A port with the mutate flag on it.
 *
 * \reqs
 *
 * \req The \port{number} port must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT mutate_process
  : public process
{
public:
  PLUGIN_INFO( "mutate",
               "A process with a mutable flag" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  mutate_process(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~mutate_process();

protected:
  /**
   * \brief Step the process.
   */
  void _step() override;

private:
  class priv;
  std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_MUTATE_PROCESS_H
