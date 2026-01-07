/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Process query descriptors using IQR and SVM ranking
 */

#ifndef VIAME_SVM_PROCESS_QUERY_PROCESS_H
#define VIAME_SVM_PROCESS_QUERY_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_svm_export.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <memory>

namespace viame
{

namespace svm
{

// -----------------------------------------------------------------------------
/**
 * @brief Process query descriptors using IQR and SVM ranking
 *
 * This process implements Interactive Query Refinement (IQR) for descriptor-
 * based search and ranking. It takes positive and negative example descriptors,
 * trains an SVM model, and returns ranked results from a descriptor index.
 *
 * This process provides IQR functionality using libSVM.
 *
 * Features:
 * - Nearest-neighbor based working index construction
 * - SVM training with probability estimates
 * - Ranked results based on SVM decision scores
 * - Feedback descriptors for active learning
 * - Serializable SVM model support
 */
class VIAME_PROCESSES_SVM_NO_EXPORT process_query_process
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  process_query_process( config_block_sptr const& config );
  virtual ~process_query_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class process_query_process

} // end namespace svm
} // end namespace viame

#endif // VIAME_SVM_PROCESS_QUERY_PROCESS_H
