// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_SHARED_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_SHARED_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file shared_process.h
 *
 * \brief Declaration of the shared process.
 */

namespace sprokit
{

/**
 * \class shared_process
 *
 * \brief A process with a shared output port.
 *
 * \process A process with a shared output port.
 *
 * \oports
 *
 * \oport{shared} A shared datum.
 *
 * \reqs
 *
 * \req The \port{shared} output must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT shared_process
  : public process
{
public:
  PLUGIN_INFO( "shared",
               "A process with the shared flag" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  shared_process(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~shared_process();

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

#endif // SPROKIT_PROCESSES_EXAMPLES_SHARED_PROCESS_H
