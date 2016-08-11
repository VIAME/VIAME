/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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

#include <vital/vital_config.h>
#include <vital/vital_export.h>

#include <string>
#include <vector>
#include <memory>

#include <vital/config/config_block.h>
#include <vital/registrar.h>
#include <vital/logger/logger.h>

namespace kwiver {
namespace vital {

/// Forward declaration of algorithm
class algorithm;
/// Shared pointer to an algorithm
typedef std::shared_ptr< algorithm > algorithm_sptr;

/// An abstract base class for all algorithms
class VITAL_EXPORT algorithm
{
public:
  virtual ~algorithm() VITAL_DEFAULT_DTOR;

  /// Return the name of this algorithm
  virtual std::string type_name() const = 0;

  /// Return the name of this implementation
  virtual std::string impl_name() const = 0;

  /// Returns an optional descriptive string for an implementation
  virtual std::string description() const = 0;

  /// Returns a clone of this algorithm as a base class shared pointer
  virtual algorithm_sptr base_clone() const = 0;

  /// Factory method to make an instance of an algorithm by type_name and impl_name
  static algorithm_sptr create(std::string const& type_name,
                               std::string const& impl_name);

  /// Return a vector of the impl_name of each registered implementation for type_name
  /**
   * \param type_name Type name of algorithm for which to find all implementation names
   *
   * \note If type_name is not specified or is an empty string, this function
   *       will return the all registered algorithms of any type.  The returned
   *       names will be of the form "type_name:impl_name".  If type_name is
   *       specified, the results will be of the form "impl_name".
   */
  static std::vector<std::string> registered_names(std::string const& type_name = "");

  /// Check the given type name against registered algorithms
  /**
   * \param type_name Type name of algorithm to validate
   * \returns true if the given \c type_name describes at least one valid
   *          registered algorithm, or false if not.
   *
   * \note This algorithm returns false for an valid abstract type name if there
   *       are no concrete instances registered with the plugin manager
   */
  static bool has_type_name(std::string const& type_name);

  /// Check the given type and implementation names against registered algorithms
  /**
   * \param type_name Type name of algorithm to validate
   * \param impl_name Implementation name of algorithm to validate
   * \returns true if the given \c type_name and \c impl_name describe a valid
   *          registered algorithm, or false if not.
   */
  static bool has_impl_name(std::string const& type_name,
                            std::string const& impl_name);


  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  /**
   * This base virtual function implementation returns an empty configuration
   * block whose name is set to \c this->type_name.
   *
   * \returns \c config_block containing the configuration for this algorithm
   *          and any nested components.
   */
  virtual config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  /**
   * \throws no_such_configuration_value_exception
   *    Thrown if an expected configuration value is not present.
   * \throws algorithm_configuration_exception
   *    Thrown when the algorithm is given an invalid \c config_block or is'
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
   * configured and returned via the \c nested_algo pointer.
   *
   * The nested algorithm will not be set if the implementation switch (as
   * defined in the \c get_nested_algo_configuration) is not present or set to
   * an invalid value relative to the registered names for this
   * \c type_name
   *
   * \param[in] type_name           The type name of the nested algorithm.
   * \param[in] name                An identifying name for the nested algorithm.
   * \param[in] config              The \c config_block instance from which we will
   *                                draw configuration needed for the nested
   *                                algorithm instance.
   * \param[in,out] nested_algo The nested algorithm's sptr variable.
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
   * that one of the derived classes will attach a more meaningful
   * logger.
   *
   * \param name Name of the logger to attach.
   */
  void attach_logger( std::string const& name );


  /**
   * \brief Logger handle.
   *
   * This handle supplies a logger for all derived classes.
   */
  kwiver::vital::logger_handle_t m_logger;

};


// ------------------------------------------------------------------
/// An intermediate templated base class for algorithm definition
/**
 *  Uses the curiously recurring template pattern to declare the
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
class VITAL_EXPORT algorithm_def
  : public algorithm
{
public:
  /// Shared pointer type of the templated vital::algorithm_def class
  typedef std::shared_ptr<Self> base_sptr;

  virtual ~algorithm_def() VITAL_DEFAULT_DTOR;

  /// Returns a clone of this algorithm as a specific algorithm pointer
  virtual base_sptr clone() const = 0;

  /// Register instances of this algorithm with a given registrar
  static bool register_instance(registrar &reg, base_sptr inst);

  /// Factory method to make an instance of this algorithm by impl_name
  static base_sptr create(std::string const& impl_name);

  /// Return a vector of the impl_name of each registered implementation
  static std::vector<std::string> registered_names();

  /// Return the name of this algorithm.
  virtual std::string type_name() const { return Self::static_type_name(); }

  /// Check the given name against registered implementation names
  /**
   * If the given name is not a valid implementation name.
   *
   * \param impl_name implementation name to check for the validity of.
   * \returns true if the given \c impl_name is valid for this definition, or
   *          false if it is not a valid implementation name.
   */
  static bool has_impl_name(std::string const& impl_name);

  /// Return a \c config_block for all registered implementations
  /**
   * For each registered implementation of this definition, add a subblock,
   * labeled under the name of that implementation, of that implementation's
   * config_block.
   *
   * If no implementation of this definition has any configuraiton properties,
   * the \c config_block returned will be empty.
   */
  static config_block_sptr get_impl_configurations();

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

  /// Instantiate nested algorighm.
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
   * \param[in] name              An identifying name for the nested algorithm.
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


// ------------------------------------------------------------------
/// An intermediate templated base class for algorithm implementations
/**
 *  Uses a variation of the curiously recurring template pattern (CRTP) to
 *  implement the clone() and register_self() functions for the derived class
 *  with respect to the base algorithm_def implementation.
 *
 *  Each algorithm implementation should be declared as shown below
 *  \code
    class my_algo_impl
    : public algorithm_impl<my_algo_impl, my_algo_def>
    {
      ...
    };
    \endcode
 *  where \c my_algo_def is the abstract algorithm class being implemented.
 *
 *  While the above example shows the algorithm_impl taking two template
 *  arguments, the full definition takes 3: the implementation class, the parent
 *  definition class that's being sub-classed and the base definition class. We
 *  define here that the "base" definition class is the first class along a
 *  class hierarchy that implements the algorithm_def class (e.g. the
 *  detect_features class or the estimate_homography class defined in
 *  kwiver::vital::algo)
 *
 *  In most cases, the parent definition class *is* the base definition class.
 *  In some cases, however, algorithm_def implementations may be sub-classed,
 *  giving rise to intermediate definitions. These cannot just directly be used
 *  as the parent *and* based because of the use of CRTP and templated types
 *  conflicting in one or more parts of the hierarchy. Thus, if an algorithm
 *  implementation is using a parent definition that is NOT the base definition,
 *  the base must also be provided as the third template argument. It also
 *  follows that the Parent algorithm_def type provided must descend from the
 *  BaseDef algorithm_def type provided. IF this is not the case, some compilers
 *  may error if this is the case (e.g. GCC will fail with an "invalid covariant
 *  return type" due to the algorithm_impl's clone method).
 *
 *  \sa algorithm_def
 */
template <typename Self, typename Parent, typename BaseDef=Parent>
class algorithm_impl
  : public Parent
{
public:
  /// shared pointer type of this impl's base vital::algorithm_def class.
  typedef std::shared_ptr<BaseDef> base_sptr;

  virtual ~algorithm_impl() VITAL_DEFAULT_DTOR;

  /// Returns a clone of this algorithm as a generic algorithm pointer
  virtual algorithm_sptr base_clone() const
  {
    return algorithm_sptr(new Self(static_cast<const Self&>(*this)));
  }

  /// Returns a clone of this algorithm as a pointer of the base definition type
  virtual base_sptr clone() const
  {
    return base_sptr(new Self(static_cast<const Self&>(*this)));
  }

  /// Return an optional descriptive string for an implementation
  virtual std::string description() const
  {
    return "";
  }

  /// Register this algorithm implementation under the BaseDef type definition
  static bool register_self(registrar &reg = vital::registrar::instance())
  {
    return algorithm_def<BaseDef>::register_instance(reg, base_sptr(new Self));
  }

};


} // end namespace vital
} // end namespace kwiver

#endif // VITAL_ALGO_ALGORITHM_H_
