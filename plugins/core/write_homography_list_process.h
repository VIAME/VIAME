/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Custom VIAME homography list writer
 */

#ifndef VIAME_CORE_WRITE_HOMOGRAPHY_LIST_H
#define VIAME_CORE_WRITE_HOMOGRAPHY_LIST_H

#include <sprokit/pipeline/process.h>

#include "viame_processes_core_export.h"

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Register optical and thermal imagery using core
 */
class VIAME_PROCESSES_CORE_NO_EXPORT write_homography_list_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  write_homography_list_process( kwiver::vital::config_block_sptr const& config );
  virtual ~write_homography_list_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class write_homography_list_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_WRITE_HOMOGRAPHY_LIST_H
