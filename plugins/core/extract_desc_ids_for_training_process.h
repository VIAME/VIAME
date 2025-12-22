/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Extract descriptor IDs overlapping with groundtruth
 */

#ifndef VIAME_CORE_EXTRACT_DESC_IDS_FOR_TRAINING_PROCESS_H
#define VIAME_CORE_EXTRACT_DESC_IDS_FOR_TRAINING_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Extract descriptor IDs overlapping with groundtruth
 *
 * This process extracts descriptor IDs stored in some database or data store
 * for later model training.
 *
 * Currently the only thing it is used for is training SVM models without user
 * interaction.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT extract_desc_ids_for_training_process
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  extract_desc_ids_for_training_process( config_block_sptr const& config );
  virtual ~extract_desc_ids_for_training_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class extract_desc_ids_for_training_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_EXTRACT_DESC_IDS_FOR_TRAINING_PROCESS_H
