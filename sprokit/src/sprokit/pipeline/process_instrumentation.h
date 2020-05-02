/*ckwg +29
 * Copyright 2016-2019 by Kitware, Inc.
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
 * \brief Interface for process instrumentation.
 */

#ifndef SPROKIT_PROCESS_INSTRUMENTATION_H
#define SPROKIT_PROCESS_INSTRUMENTATION_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include <vital/config/config_block.h>

#include <string>

namespace sprokit {

class process; // incomplete type


// -----------------------------------------------------------------
/**
 * \brief Base class for process instrumentation.
 *
 * This class is the abstract base class for process instrumentation.
 * It defines the interface processes can use to access an
 * instrumentation package using the strategy pattern.
 */
class SPROKIT_PIPELINE_EXPORT process_instrumentation
{
public:
  process_instrumentation();
  virtual ~process_instrumentation() = default;

  void set_process( sprokit::process const& proc );

  virtual void start_init_processing( std::string const& data ) = 0;
  virtual void stop_init_processing() = 0;

  virtual void start_finalize_processing( std::string const& data ) = 0;
  virtual void stop_finalize_processing() = 0;

  virtual void start_reset_processing( std::string const& data ) = 0;
  virtual void stop_reset_processing() = 0;

  virtual void start_flush_processing( std::string const& data ) = 0;
  virtual void stop_flush_processing() = 0;

  virtual void start_step_processing( std::string const& data ) = 0;
  virtual void stop_step_processing() = 0;

  virtual void start_configure_processing( std::string const& data ) = 0;
  virtual void stop_configure_processing() = 0;

  virtual void start_reconfigure_processing( std::string const& data ) = 0;
  virtual void stop_reconfigure_processing() = 0;


  /**
   * @brief Get process that is being instrumented.
   *
   *
   * @return Pointer to process object.
   */
  sprokit::process const* process() const { return m_process; }

  /**
   * @brief Get name or process
   *
   * This method returns the name of the associated process.
   *
   * @return Name of process.
   */
  std::string process_name() const;

  /**
   * @brief Configure provider.
   *
   * This method sends the config block to the implementation.
   *
   * @param conf Configuration block.
   */
  virtual void configure( kwiver::vital::config_block_sptr const conf );

  /**
   * @brief Get default configuration block.
   *
   * This method returns the default configuration block for this
   * instrumentation implementation.
   *
   * @return Pointer to config block.
   */
  virtual kwiver::vital::config_block_sptr get_configuration() const;

private:
  sprokit::process const* m_process;
}; // end class process_instrumentation

} // end namespace

#endif /* SPROKIT_PROCESS_INSTRUMENTATION_H */
