/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Pass frame with step index and in min max limits
 */

#ifndef VIAME_CORE_FILTER_FRAME_INDEX_PROCESS_H
#define VIAME_CORE_FILTER_FRAME_INDEX_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Pass frame with step index and in min max limits
 */
class VIAME_PROCESSES_CORE_NO_EXPORT filter_frame_index_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  filter_frame_index_process( kwiver::vital::config_block_sptr const& config );
  virtual ~filter_frame_index_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class filter_frame_index_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_FILTER_FRAME_INDEX_PROCESS_H
