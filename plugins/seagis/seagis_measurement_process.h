/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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
