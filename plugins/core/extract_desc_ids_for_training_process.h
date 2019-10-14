/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
