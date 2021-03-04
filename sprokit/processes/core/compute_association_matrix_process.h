// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_COMPUTE_ASSOCIATION_MATRIX_PROCESS_H_
#define _KWIVER_COMPUTE_ASSOCIATION_MATRIX_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

#include <memory>

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * \class compute_association_matrix_process
 *
 * \brief Generates association matrix between old tracks and new detections
 *        for use in object tracking.
 *
 * \iports
 * \iport{timestamp}
 * \iport{image}
 * \iport{tracks}
 * \iport{detections}
 *
 * \oports
 * \oport{matrix_d}
 * \oport{tracks}
 * \oport{detections}
 */
class KWIVER_PROCESSES_NO_EXPORT compute_association_matrix_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "compute_association_matrix",
               "Computes cost matrix for adding new detections to existing tracks." )

  compute_association_matrix_process( vital::config_block_sptr const& config );
  virtual ~compute_association_matrix_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class compute_association_matrix_process

} // end namespace
#endif /* _KWIVER_COMPUTE_ASSOCIATION_MATRIX_PROCESS_H_ */
