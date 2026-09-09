// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief base algorithm/_def/_impl class interfaces

// based on kwiver/vital/algo/algorithm.h
// @7e4920796821476afb9be3b31b23791a1e1e76b6

#ifndef VITAL_ALGO_ALGORITHM_H_
#define VITAL_ALGO_ALGORITHM_H_

#include <viame/algorithm_framework/algo/vital_algo_export.h>
#include <viame/algorithm_framework/vital_config.h>
#include <viame/algorithm_framework/vital_export.h>

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/plugin/pluggable.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>
#include <viame/algorithm_framework/plugin/plugin_info.h>

#include <viame/algorithm_framework/config/config_helpers.txx>

#include <memory>
#include <string>
#include <vector>

namespace kwiver {

namespace vital {

/// Forward declaration of algorithm
class algorithm;
/// Shared pointer to an algorithm
typedef std::shared_ptr< algorithm > algorithm_sptr;

// ----------------------------------------------------------------------------

/// @brief An abstract base class for all algorithms
///
/// This class is an abstract base class for all algorithm
/// implementations.
class VITAL_ALGO_EXPORT algorithm :  public vital::pluggable
{
public:
  algorithm();
  PLUGGABLE_INTERFACE( algorithm );

  /// Get \link kwiver::vital::config_block configuration
  /// block \endlink holding the default configuration values for this class.
  ///
  /// This base function implementation returns the config block unmodified.
  ///
  /// \returns \c config_block containing the configuration for this algorithm
  ///          and any nested components.
  static void get_default_config(
    [[maybe_unused]] ::kwiver::vital::config_block& cb );

  /// Set this algorithm's properties via a config block
  ///
  /// This method is called to pass a configuration to the
  /// algorithm. The implementation of this method should be
  /// light-weight and only save the necessary config values. Defer
  /// any substantial processing in another method.
  ///
  /// \throws no_such_configuration_value_exception
  ///    Thrown if an expected configuration value is not present.
  ///
  /// \throws algorithm_configuration_exception
  ///    Thrown when the algorithm is given an invalid \c config_block or is
  ///    otherwise unable to configure itself.
  ///
  /// \param config  The \c config_block instance containing the configuration
  ///                parameters for this algorithm
  virtual void set_configuration( config_block_sptr config ) = 0;

  /// Get this algorithm's \link kwiver::vital::config_block configuration
  /// block \endlink
  ///
  /// This method returns the required configuration for the
  /// algorithm. The implementation of this method should be
  /// light-weight and only create and fill in the config
  /// block.
  ///
  /// This base virtual function implementation returns an empty configuration.
  ///
  /// \returns \c config_block containing the configuration for this algorithm
  ///          and any nested components.
  virtual config_block_sptr get_configuration() const;

  /// Check that the algorithm's configuration config_block is valid
  ///
  /// This checks solely within the provided \c config_block and not against
  /// the current state of the instance. This isn't static for inheritance
  /// reasons.
  ///
  /// \param config  The config block to check configuration of.
  ///
  /// \returns true if the configuration check passed and false if it didn't.
  virtual bool check_configuration( config_block_sptr config ) const = 0;

  void set_impl_name( const std::string& name );
  std::string impl_name() const;
  kwiver::vital::logger_handle_t logger() const;

protected:
  /// \brief Attach logger to this object.
  ///
  /// This method attaches a logger to this object. The name supplied
  /// is used to name the logger. Since this is a fundamental base
  /// class, derived classes will want to have the logger named
  /// something relevant to the concrete algorithm.
  ///
  /// A logger is attached by the base class, but it is expected that
  /// one of the derived classes will attach a more meaningful logger.
  ///
  /// \param name Name of the logger to attach.
  void attach_logger( std::string const& name );

  /// \brief Initialize the internals of the algorithm.
  ///
  /// This is overridden every time an algorithm want to initialize any
  /// internal
  /// state. The pluggable macros will make sure to call it in auto-generated
  /// constructor.
  virtual void initialize();

  virtual void set_configuration_internal( config_block_sptr config );

private:
  /// \brief Logger handle.
  ///
  /// This handle supplies a logger for all derived classes.
  kwiver::vital::logger_handle_t m_logger;

  std::string m_impl_name;
};

}  // namespace kwiver::vital

} // namespace kwiver

#endif // VITAL_ALGO_ALGORITHM_H_
