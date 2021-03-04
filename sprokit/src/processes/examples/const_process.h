// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_CONST_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_CONST_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file const_process.h
 *
 * \brief Declaration of the const process.
 */

namespace sprokit {

/**
 * \class const_process
 *
 * \brief A process with a const output port.
 *
 * \process A process with a const output port.
 *
 * \oports
 *
 * \oport{const} A constant datum.
 *
 * \reqs
 *
 * \req The \port{const} output must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT const_process
    : public process
{
public:
  PLUGIN_INFO( "const",
               "A process wth a const flag" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  const_process(kwiver::vital::config_block_sptr const& config);
  /**
   * \brief Destructor.
   */
  ~const_process();

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

#endif // SPROKIT_PROCESSES_EXAMPLES_CONST_PROCESS_H
