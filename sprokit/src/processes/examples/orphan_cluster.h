// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H
#define SPROKIT_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process_cluster.h>

/**
 * \file orphan_cluster.h
 *
 * \brief Declaration of the orphan cluster.
 */

namespace sprokit
{

/**
 * \class orphan_cluster
 *
 * \brief A no-op cluster.
 *
 * \process A no-op cluster.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT orphan_cluster
  : public process_cluster
{
public:
  PLUGIN_INFO( "orphan_cluster",
               "A dummy cluster" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  orphan_cluster(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  ~orphan_cluster();
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H
