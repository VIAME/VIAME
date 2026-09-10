// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   provided_by_cluster.h
 * @brief  Interface to provided_by_cluster class.
 */

#ifndef SPROKIT_PIPELINE_UTIL_PROVIDED_BY_CLUSTER_H
#define SPROKIT_PIPELINE_UTIL_PROVIDED_BY_CLUSTER_H

#include "bakery_base.h"

namespace sprokit {

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class provided_by_cluster
{
public:
  provided_by_cluster( process::type_t const& name, process::names_t const& procs );
  ~provided_by_cluster();

  bool operator()( bakery_base::config_decl_t const& decl ) const;

private:
  process::type_t const m_name;
  process::names_t const m_procs;
};

} // end namespace sprokit

#endif /* SPROKIT_PIPELINE_UTIL_PROVIDED_BY_CLUSTER_H */
