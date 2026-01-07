/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Train SVM models from descriptor index and label files
 */

#ifndef VIAME_SVM_TRAIN_SVM_MODELS_PROCESS_H
#define VIAME_SVM_TRAIN_SVM_MODELS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_svm_export.h"

#include <memory>

namespace viame
{

namespace svm
{

// -----------------------------------------------------------------------------
/**
 * @brief Train SVM models for multiple categories from descriptor index
 *
 * This process trains binary SVM classifiers for each object category using
 * libsvm. It reads descriptor vectors from a CSV index file and positive/negative
 * label files containing descriptor UIDs.
 *
 * This process provides SVM model training functionality using libSVM.
 *
 * Features:
 * - Reads descriptor index from CSV
 * - Loads positive/negative UIDs from label files per category
 * - Hard negative mining via nearest neighbor search
 * - Configurable sampling limits
 * - Outputs trained SVM models (.svm files)
 *
 * This process runs once and trains all category models, then completes.
 */
class VIAME_PROCESSES_SVM_NO_EXPORT train_svm_models_process
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  train_svm_models_process( config_block_sptr const& config );
  virtual ~train_svm_models_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class train_svm_models_process

} // end namespace svm
} // end namespace viame

#endif // VIAME_SVM_TRAIN_SVM_MODELS_PROCESS_H
