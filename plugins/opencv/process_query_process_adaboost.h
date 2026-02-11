/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Process query descriptors using IQR and AdaBoost ranking
 */

#ifndef VIAME_OPENCV_PROCESS_QUERY_PROCESS_ADABOOST_H
#define VIAME_OPENCV_PROCESS_QUERY_PROCESS_ADABOOST_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_opencv_export.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <memory>

namespace viame
{

// -----------------------------------------------------------------------------
/**
 * @brief Process query descriptors using IQR and AdaBoost ranking
 *
 * This process implements Interactive Query Refinement (IQR) for descriptor-
 * based search and ranking. It takes positive and negative example descriptors,
 * trains an AdaBoost model, and returns ranked results from a descriptor index.
 *
 * This process provides IQR functionality using OpenCV's cv::ml::Boost.
 *
 * Features:
 * - Nearest-neighbor based working index construction
 * - AdaBoost training with probability estimates via sigmoid
 * - Ranked results based on AdaBoost decision scores
 * - Feedback descriptors for active learning
 * - Serializable AdaBoost model support
 */
class VIAME_PROCESSES_OPENCV_NO_EXPORT process_query_process_adaboost
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  process_query_process_adaboost( config_block_sptr const& config );
  virtual ~process_query_process_adaboost();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class process_query_process_adaboost

} // end namespace viame

#endif // VIAME_OPENCV_PROCESS_QUERY_PROCESS_ADABOOST_H
