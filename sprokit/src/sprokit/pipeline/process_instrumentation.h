// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
