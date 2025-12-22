/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief SEAGIS stereo measurement process
 */

#ifndef VIAME_SEAGIS_MEASUREMENT_PROCESS_H
#define VIAME_SEAGIS_MEASUREMENT_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/seagis/viame_processes_seagis_export.h>

#include <memory>

namespace viame
{

namespace seagis
{

// -----------------------------------------------------------------------------
/**
 * @brief SEAGIS stereo measurement process
 *
 * Computes measurements from stereo camera track data using the SEAGIS
 * StereoLibLX library for camera calibration and 3D intersection calculations.
 */
class VIAME_PROCESSES_SEAGIS_NO_EXPORT seagis_measurement_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  seagis_measurement_process( kwiver::vital::config_block_sptr const& config );
  virtual ~seagis_measurement_process();

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

}; // end class seagis_measurement_process

} // end namespace seagis
} // end namespace viame

#endif // VIAME_SEAGIS_MEASUREMENT_PROCESS_H
