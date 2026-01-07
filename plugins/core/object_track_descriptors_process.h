/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Attach descriptors to object track states from file
 */

#ifndef VIAME_CORE_OBJECT_TRACK_DESCRIPTORS_PROCESS_H
#define VIAME_CORE_OBJECT_TRACK_DESCRIPTORS_PROCESS_H

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
 * @brief Attach descriptors to object track states from file
 *
 * This process takes in an object_track_set and attaches descriptors to each
 * track state by looking them up from a CSV file.
 *
 * The input file format is CSV: track_id,frame_id,val1,val2,...,valN
 */
class VIAME_PROCESSES_CORE_NO_EXPORT object_track_descriptors_process
  : public sprokit::process
{
public:
  using config_block_sptr = kwiver::vital::config_block_sptr;

  // -- CONSTRUCTORS --
  object_track_descriptors_process( config_block_sptr const& config );
  virtual ~object_track_descriptors_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();
  void load_descriptor_index();

  class priv;
  const std::unique_ptr< priv > d;

}; // end class object_track_descriptors_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_OBJECT_TRACK_DESCRIPTORS_PROCESS_H
