// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file tagged_flow_dependent_process.h
 *
 * \brief Declaration of the tagged flow dependent process.
 */

namespace sprokit
{

/**
 * \class tagged_flow_dependent_process
 *
 * \brief A process with tagged flow dependent ports.
 *
 * \process A process with flow dependent ports.
 *
 * \iports
 *
 * \iport{untagged_input} An untagged flow dependent input port.
 * \iport{tagged_input} A tagged flow dependent input port.
 *
 * \oports
 *
 * \oport{untagged_output} An untagged flow dependent output port.
 * \oport{tagged_output} A tagged flow dependent output port.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT tagged_flow_dependent_process
  : public process
{
public:
  PLUGIN_INFO( "tagged_flow_dependent",
               "A process with a tagged flow dependent types" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  tagged_flow_dependent_process(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~tagged_flow_dependent_process();

protected:
  /**
   * \brief Reset the process.
   */
  void _reset() override;

  /**
   * \brief Step the process.
   */
  void _step() override;

private:
  void make_ports();

  class priv;
  std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_TAGGED_FLOW_DEPENDENT_PROCESS_H
