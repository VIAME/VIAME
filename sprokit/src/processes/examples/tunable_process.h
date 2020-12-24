// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_TUNABLE_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_TUNABLE_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file tunable_process.h
 *
 * \brief Declaration of the tunable process.
 */

namespace sprokit {

/**
 * \class tunable_process
 *
 * \brief A tunable process.
 *
 * \process Outputs a tunable result.
 *
 * \oports
 *
 * \oport{tunable} The tunable output.
 * \oport{non_tunable} The non-tunable output.
 *
 * \configs
 *
 * \config{tunable} The tunable value to use.
 * \config{non_tunable} The non-tunable value to use.
 *
 * \reqs
 *
 * \req The \port{tunable} output must be connected.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT tunable_process
  : public process
{
public:
  PLUGIN_INFO( "tunable",
               "A process with a tunable parameter" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  tunable_process(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~tunable_process();

protected:
  /**
   * \brief Configure the process.
   */
  void _configure() override;

  /**
   * \brief Step the process.
   */
  void _step() override;

  /**
   * \brief Step the process.
   */
  void _reconfigure(kwiver::vital::config_block_sptr const& conf) override;

private:
  class priv;
  std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_TUNABLE_PROCESS_H
