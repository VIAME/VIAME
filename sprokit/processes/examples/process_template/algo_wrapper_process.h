// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_PROCESSES_ALGO_WRAPPER_PROCESS_H
#define ARROWS_PROCESSES_ALGO_WRAPPER_PROCESS_H

#include <sprokit/pipeline/process.h>

//+ use correct export include file
#include "kwiver_processes_export.h"

#include <vital/config/config_block.h>

namespace kwiver {

// ----------------------------------------------------------------
class KWIVER_PROCESSES_NO_EXPORT algo_wrapper_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "algo_wrapper",
               "Template process to wrap an algo" );

  algo_wrapper_process( kwiver::vital::config_block_sptr const& config );
  virtual ~algo_wrapper_process();

protected:
  virtual void _configure();
  virtual void _step();
  //+ Implement other process base class methods as needed

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace

#endif // ARROWS_PROCESSES_ALGO_WRAPPER_PROCESS_H
