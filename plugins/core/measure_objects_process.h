/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo measurement process
 */

#ifndef VIAME_CORE_MEASURE_OBJECTS_PROCESS_H
#define VIAME_CORE_MEASURE_OBJECTS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Stereo measurement process
 *
 * Computes measurements from stereo camera track data by finding corresponding
 * points between left and right camera views and triangulating to get 3D lengths.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT measure_objects_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  measure_objects_process( kwiver::vital::config_block_sptr const& config );
  virtual ~measure_objects_process();

protected:
  virtual void _configure();
  virtual void _step();
  virtual void _init();

  void input_port_undefined( port_t const& port ) override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class measure_objects_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_MEASURE_OBJECTS_PROCESS_H
