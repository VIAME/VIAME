// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_TEST_PROC_SEQ_H
#define SPROKIT_PROCESSES_TEST_PROC_SEQ_H

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_test_export.h"

namespace kwiver {

class KWIVER_PROCESSES_TEST_NO_EXPORT test_proc_seq
  : public sprokit::process
{
public:
  PLUGIN_INFO( "test_proc_seq",
               "Test method sequence in processes" )

  test_proc_seq( kwiver::vital::config_block_sptr const& config );

  /**
   * \brief Destructor.
   */
  virtual ~test_proc_seq() = default;

protected:
  void _configure() override;
  void _init() override;
  void _step() override;
  void _finalize() override;
};

}

#endif // SPROKIT_PROCESSES_TEST_PROC_SEQ_H
