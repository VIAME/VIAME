// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_EXPECT_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_EXPECT_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file expect_process.h
 *
 * \brief Declaration of the expect process.
 */

namespace sprokit
{

/**
 * \class expect_process
 *
 * \brief A process which checks values.
 *
 * \process A process which checks values.
 *
 * \oports
 *
 * \oport{dummy} A dummy port.
 *
 * \configs
 *
 * \config{tunable} A tunable parameter.
 * \config{expect} The expected string.
 * \config{expect_key} Whether to expect a key or value.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT expect_process
  : public process
{
public:
  PLUGIN_INFO( "expect",
               "A process which expects some conditions" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  expect_process(kwiver::vital::config_block_sptr const& config);
  /**
   * \brief Destructor.
   */
  ~expect_process();

protected:
  void _configure() override;
  void _step() override;
  void _reconfigure(kwiver::vital::config_block_sptr const& conf) override;

private:
  class priv;
  std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_EXPECT_PROCESS_H
