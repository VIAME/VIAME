/*ckwg +29
 * Copyright 2013-2017, 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief base algorithm/_def/_impl class interfaces
 */

#ifndef VITAL_ALGO_ALGORITHM_H_
#define VITAL_ALGO_ALGORITHM_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/algo/vital_algo_export.h>

#include <vital/config/config_block.h>
#include <vital/logger/logger.h>
#include <vital/plugin_loader/plugin_info.h>

#include <string>
#include <vector>
#include <memory>

namespace kwiver {
namespace vital {

/// Forward declaration of algorithm
class algorithm;
/// Shared pointer to an algorithm
typedef std::shared_ptr< algorithm > algorithm_sptr;

// ----------------------------------------------------------------
/**
 * @brief An abstract base class for all algorithms
 *
 * This class is an abstract base class for all algorithm
 * implementations.
 */
class VITAL_ALGO_EXPORT algorithm
{
public:
  virtual ~algorithm() = default;

  /// Return the name of the base algorithm
  virtual std::string type_name() const = 0;

  /// Return the name of this implementation
  virtual std::string impl_name() const final;

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  /**
   * This method returns the required configuration for the
   * algorithm. The implementation of this method should be
   * light-weight and only create and fill in the config
   * block.
   *
   * This base virtual function implementation returns an empty configuration.
   *
   * \returns \c config_block containing the configuration for this algorithm
   *          and any nested components.
   */
  virtual config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  /**
   * This method is called to pass a configuration to the
   * algorithm. The implementation of this method should be
   * light-weight and only save the necessary config values. Defer
   * any substantial processing in another method.
   *
   * \throws no_such_configuration_value_exception
   *    Thrown if an expected configuration value is not present.
   *
   * \throws algorithm_configuration_exception
   *    Thrown when the algorithm is given an invalid \c config_block or is
   *    otherwise unable to configure itself.
   *
   * \param config  The \c config_block instance containing the configuration
   *                parameters for this algorithm
   */
  virtual void set_configuration(config_block_sptr config) = 0;

  /// Check that the algorithm's configuration config_block is valid
  /**
   * This checks solely within the provided \c config_block and not against
   * the current state of the instance. This isn't static for inheritance
   * reasons.
   *
   * \param config  The config block to check configuration of.
   *
   * \returns true if the configuration check passed and false if it didn't.
   */
  virtual bool check_configuration(config_block_sptr config) const = 0;

  /// Helper function for properly getting a nested algorithm's configuration
  /**
   * Adds a configurable algorithm implementation switch for this algorithm.
   * If the variable pointed to by \c nested_algo is a defined sptr to an
   * implementation, its \link kwiver::vital::config_block configuration \endlink
   * parameters are merged with the given
   * \link kwiver::vital::config_block config_block \endlink.
   *
   * \param[in]       type_name   The type name of the nested algorithm.
   * \param[in]       name        An identifying name for the nested algorithm
   * \param[in,out]   config      The \c config_block instance in which to put the
   *                              nested algorithm's configuration.
   * \param[in]       nested_algo The nested algorithm's sptr variable.
   */
  static void get_nested_algo_configuration(std::string const& type_name,
                                            std::string const& name,
                                            config_block_sptr config,
                                            algorithm_sptr nested_algo);

  /// Helper function for properly setting a nested algorithm's configuration
  /**
   * If the value for the config parameter "type" is supported by the
   * concrete algorithm class, then a new algorithm object is created,
   * configured using the set_configuration() method and returned via
   * the \c nested_algo pointer.
   *
   * The nested algorithm will not be set if the implementation type (as
   * defined in the \c get_nested_algo_configuration) is not present or set to
   * an invalid value relative to the registered names for this
   * \c type_name
   *
   * \param[in] type_name           The type name of the nested algorithm.
   * \param[in] name                Config block name for the nested algorithm.
   * \param[in] config              The \c config_block instance from which we will
   *                                draw configuration needed for the nested
   *                                algorithm instance.
   * \param[out] nested_algo The nested algorithm's sptr variable.
   */
  static void set_nested_algo_configuration(std::string const& type_name,
                                            std::string const& name,
                                            config_block_sptr config,
                                            algorithm_sptr &nested_algo);

  /// Helper function for checking that basic nested algorithm configuration is valid
  /**
   * Check that the expected implementation switch exists and that its value is
   * registered implementation name.
   *
   * If the name is valid, we also recursively call check_configuration() on the
   * set implementation. This is done with a fresh create so we don't have to
   * rely on the implementation being defined in the instance this is called
   * from.
   *
   * \param     type_name   The type name of the nested algorithm.
   * \param     name        An identifying name for the nested algorithm.
   * \param     config  The \c config_block to check.
   */
  static bool check_nested_algo_configuration(std::string const& type_name,
                                              std::string const& name,
                                              config_block_sptr config);

  void set_impl_name( const std::string& name );
  kwiver::vital::logger_handle_t logger() const;

protected:
  algorithm(); // CTOR

  /**
   * \brief Attach logger to this object.
   *
   * This method attaches a logger to this object. The name supplied
   * is used to name the logger. Since this is a fundamental base
   * class, derived classes will want to have the logger named
   * something relevant to the concrete algorithm.
   *
   * A logger is attached by the base class, but it is expected that
   * one of the derived classes will attach a more meaningful logger.
   *
   * \param name Name of the logger to attach.
   */
  void attach_logger( std::string const& name );

private:
  /**
   * \brief Logger handle.
   *
   * This handle supplies a logger for all derived classes.
   */
  kwiver::vital::logger_handle_t m_logger;

  std::string m_impl_name;
};


// ------------------------------------------------------------------
/// An intermediate templated base class for algorithm definition
/**
 *  Uses the curiously recurring template pattern (CRTP) to declare the
 *  clone function and automatically provide functions to register
 *  algorithm, and create new instance by name.
 *  Each algorithm definition should be declared as shown below
 *  \code
    class my_algo_def
    : public algorithm_def<my_algo_def>
    {
      ...
    };
    \endcode
 *  \sa algorithm_impl
 */
template <typename Self>
class VITAL_ALGO_EXPORT algorithm_def
  : public algorithm
{
public:
  /// Shared pointer type of the templated vital::algorithm_def class
  typedef std::shared_ptr<Self> base_sptr;

  virtual ~algorithm_def() = default;

  /// Factory method to make an instance of this algorithm by impl_name
  static base_sptr create(std::string const& impl_name);

  /// Return a vector of the impl_name of each registered implementation
  static std::vector<std::string> registered_names();

  /// Return the name of this algorithm.
  virtual std::string type_name() const { return Self::static_type_name(); }

  /// Helper function for properly getting a nested algorithm's configuration
  /**
   * Adds a configurable algorithm implementation switch for this algorithm_def.
   * If the variable pointed to by \c nested_algo is a defined sptr to an
   * implementation, its \link kwiver::vital::config_block configuration \endlink
   * parameters are merged with the given
   * \link kwiver::vital::config_block config_block \endlink.
   *
   * \param     name        An identifying name for the nested algorithm
   * \param     config      The \c config_block instance in which to put the
   *                          nested algorithm's configuration.
   * \param     nested_algo The nested algorithm's sptr variable.
   */
  static void get_nested_algo_configuration(std::string const& name,
                                            config_block_sptr config,
                                            base_sptr nested_algo);

  /// Instantiate nested algorithm.
  /**
   * A new concrete algorithm object is created if the value for the
   * config parameter "type" is supported. The new object is returned
   * through the nested_algo parameter.
   *
   * The nested algorithm will not be set if the implementation switch (as
   * defined in the \c get_nested_algo_configuration) is not present or set to
   * an invalid value relative to the registered names for this
   * \c algorithm_def.
   *
   * \param[in] name              Config block name for the nested algorithm.
   * \param[in] config            The \c config_block instance from which we will
   *                              draw configuration needed for the nested
   *                              algorithm instance.
   * \param[out] nested_algo      Pointer to the algorithm object is returned here.
   */
  static void set_nested_algo_configuration(std::string const& name,
                                            config_block_sptr config,
                                            base_sptr &nested_algo);

  /// Helper function for checking that basic nested algorithm configuration is valid
  /**
   * Check that the expected implementation switch exists and that its value is
   * registered implementation name.
   *
   * If the name is valid, we also recursively call check_configuration() on the
   * set implementation. This is done with a fresh create so we don't have to
   * rely on the implementation being defined in the instance this is called
   * from.
   *
   * \param     name        An identifying name for the nested algorithm.
   * \param     config      The \c config_block to check.
   */
  static bool check_nested_algo_configuration(std::string const& name,
                                              config_block_sptr config);
};

#if 01

template <typename Self, typename Parent, typename BaseDef=Parent>
class VITAL_DEPRECATED algorithm_impl
  : public Parent
{
public:
  /// shared pointer type of this impl's base vital::algorithm_def class.
 VITAL_DEPRECATED algorithm_impl() { }
  virtual VITAL_DEPRECATED ~algorithm_impl() = default;
};

#endif

} // end namespace vital
} // end namespace kwiver

#endif // VITAL_ALGO_ALGORITHM_H_
