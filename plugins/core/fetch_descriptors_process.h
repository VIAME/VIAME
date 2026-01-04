/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Fetch descriptors from file given UIDs
 */

#ifndef VIAME_CORE_FETCH_DESCRIPTORS_PROCESS_H
#define VIAME_CORE_FETCH_DESCRIPTORS_PROCESS_H

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
 * @brief Fetch descriptors from file given UIDs
 *
 * This process takes in a vector of UIDs and fetches the corresponding
 * descriptors from a CSV file. This is a C++ replacement for the SMQTK-based
 * smqtk_fetch_descriptors Python process.
 *
 * The input file format is CSV: uid,val1,val2,...,valN (one descriptor per line)
 */
class VIAME_PROCESSES_CORE_NO_EXPORT fetch_descriptors_process
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  fetch_descriptors_process( config_block_sptr const& config );
  virtual ~fetch_descriptors_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();
  void load_descriptor_index();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class fetch_descriptors_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_FETCH_DESCRIPTORS_PROCESS_H
