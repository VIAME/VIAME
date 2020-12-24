// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_SINK_PROCESS_H
#define SPROKIT_PROCESSES_SINK_PROCESS_H

#include "processes_flow_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file sink_process.h
 *
 * \brief Declaration of the sink process.
 */

namespace sprokit
{

/**
 * \class sink_process
 *
 * \brief A process for doing nothing with a data stream.
 *
 * \process Ignores incoming data.
 *
 * \iports
 *
 * \iport{sink} The data to ignore.
 *
 * \reqs
 *
 * \req The \port{sink} port must be connected.
 *
 * \ingroup process_flow
 */
class PROCESSES_FLOW_NO_EXPORT sink_process
  : public process
{
public:
  PLUGIN_INFO( "sink",
               "Ignores incoming data." )

  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  sink_process( kwiver::vital::config_block_sptr const& config );
  /**
   * \brief Destructor.
   */
  ~sink_process();

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

#endif // SPROKIT_PROCESSES_SINK_PROCESS_H
