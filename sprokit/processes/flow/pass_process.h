// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_PASS_PROCESS_H
#define SPROKIT_PROCESSES_PASS_PROCESS_H

#include "processes_flow_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file pass_process.h
 *
 * \brief Declaration of the pass process.
 */

namespace sprokit
{

/**
 * \class pass_process
 *
 * \brief A process to pass through a data stream.
 *
 * \process Passes through incoming data.
 *
 * \iports
 *
 * \iport{pass} The datum to pass.
 *
 * \oports
 *
 * \oport{pass} The passed datum.
 *
 * \reqs
 *
 * \req The \port{pass} ports must be connected.
 *
 * \ingroup process_flow
 */
class PROCESSES_FLOW_NO_EXPORT pass_process
  : public process
{
public:
  PLUGIN_INFO( "pass",
               "Pass a data stream through." )

/**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  pass_process( kwiver::vital::config_block_sptr const& config );
  /**
   * \brief Destructor.
   */
  ~pass_process();

protected:
  /**
   * \brief Step the process.
   */
  void _step();

private:
  class priv;
  std::unique_ptr< priv > d;
};

}

#endif // SPROKIT_PROCESSES_PASS_PROCESS_H
