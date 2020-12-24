// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to template process.
 */

#ifndef KWIVER_MATLAB_PROCESS_H_
#define KWIVER_MATLAB_PROCESS_H_

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_matlab_export.h"

#include <memory>

namespace kwiver {
namespace matlab {

// ----------------------------------------------------------------
/**
 * @brief brief description
 *
 */
class KWIVER_PROCESSES_MATLAB_NO_EXPORT matlab_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "matlab_bridge",
               "Bridge to process written in matlab." )

  matlab_process( kwiver::vital::config_block_sptr const& config );
  virtual ~matlab_process();

protected:
  virtual void _configure();
  virtual void _step();
  virtual void _init();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class

} } // end namespace

#endif
