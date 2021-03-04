// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_DOWNSAMPLE_PROCESS_H_
#define _KWIVER_DOWNSAMPLE_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

#include <memory>

namespace kwiver
{

class KWIVER_PROCESSES_NO_EXPORT downsample_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "downsample",
               "Downsample an input stream." )

  downsample_process( vital::config_block_sptr const& config );
  virtual ~downsample_process();

protected:
  virtual void _configure();
  virtual void _step();
  virtual void _init();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
};

}

#endif // _KWIVER_DOWNSAMPLE_PROCESS_H_
