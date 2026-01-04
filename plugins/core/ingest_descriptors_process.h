/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Ingest descriptors from a pipeline and write to file
 */

#ifndef VIAME_CORE_INGEST_DESCRIPTORS_PROCESS_H
#define VIAME_CORE_INGEST_DESCRIPTORS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Ingest descriptors and write them to file
 *
 * This process takes in descriptor sets and matching UIDs, buffers them,
 * and writes them out to a file in batches. This is a C++ replacement for
 * the SMQTK-based smqtk_ingest_descriptors Python process.
 *
 * Features:
 * - Buffering by frame count or descriptor count for efficient batch writes
 * - CSV output format with UID and descriptor vector
 * - Pass-through of inputs to outputs for pipeline chaining
 */
class VIAME_PROCESSES_CORE_NO_EXPORT ingest_descriptors_process
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  ingest_descriptors_process( config_block_sptr const& config );
  virtual ~ingest_descriptors_process();

protected:
  virtual void _configure();
  virtual void _step();
  virtual void _finalize();

private:
  void make_ports();
  void make_config();
  void flush_buffer();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class ingest_descriptors_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_INGEST_DESCRIPTORS_PROCESS_H
