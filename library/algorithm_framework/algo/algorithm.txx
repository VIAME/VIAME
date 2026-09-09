// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief templated helpers for creating algorithms

#ifndef VITAL_ALGO_ALGORITHM_TXX_
#define VITAL_ALGO_ALGORITHM_TXX_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/config/config_helpers.txx>
#include <viame/algorithm_framework/plugin/plugin_manager.h>

#include <type_traits>

namespace kwiver::vital {

/// \brief Create algorithm from interface type and implementation name.
///
/// \tparam INTERFACE Name of the interface
/// \param impl_name Name if the implementation
///
/// \return New algorithm object or
template < typename INTERFACE >
std::shared_ptr< INTERFACE >
create_algorithm( std::string const& implementation_name )
{
  auto fact =
    implementation_factory_by_name< INTERFACE >();
  auto cb_empty = config_block::empty_config();
  kwiver::vital::plugin_manager& vpm =
    kwiver::vital::plugin_manager::instance();

  vpm.load_all_plugins();

  auto inst = fact.create( implementation_name, cb_empty );
  // TODO ideally we should avoid this
  inst->set_impl_name( implementation_name );
  return inst;
}

/// \brief Check the given type and implementation names against registered
/// algorithms.
///
/// \tparam INTERFACE name of interfacace to validate
/// \param impl_name Implementation name of algorithm to validate
/// \returns true if the given \c type_name and \c impl_name describe a valid
///          registered algorithm, or false if not.
template < typename INTERFACE >
bool
has_algorithm_impl_name( std::string const& implementation_name )
{
  // Get list of factories for the algo_name
  kwiver::vital::plugin_manager& vpm =
    kwiver::vital::plugin_manager::instance();

  auto impl_names = vpm.impl_names< INTERFACE >();

  return std::find(
    impl_names.begin(), impl_names.end(),
    implementation_name ) != impl_names.end();
}

// ----------------------------------------------------------------------------
/// Return a vector of the impl_name of each registered implementation
template < typename INTERFACE >
std::vector< std::string >
registered_names()
{
  // Get list of factories for the algo_name
  kwiver::vital::plugin_manager& vpm =
    kwiver::vital::plugin_manager::instance();

  auto impl_names = vpm.impl_names< INTERFACE >();

  return impl_names;
}

//// ----------------------------------------------------------------------------
/// Helper function for properly setting a nested algorithm's configuration
///
/// If the value for the config parameter "type" is supported by the
/// concrete algorithm class, then a new algorithm object is created,
/// configured using the set_configuration() method and returned via
/// the \c nested_algo pointer.
///
/// The nested algorithm will not be set if the implementation type (as
/// defined in the \c get_nested_algo_configuration) is not present or set to
/// an invalid value relative to the registered names for this
/// \c type_name
///
/// \tparam    INTERFACE           interface that the nested_algo implements.
/// \param[in] name                Config block name for the nested algorithm.
/// \param[in] config              The \c config_block instance from which we
/// will
///                                draw configuration needed for the nested
///                                algorithm instance.
/// \param[out] nested_algo The nested algorithm's sptr variable.
template < typename INTERFACE >
void
set_nested_algo_configuration(
  std::string const& name,
  config_block_sptr config,
  std::shared_ptr< INTERFACE >&    nested_algo )
{
  static kwiver::vital::logger_handle_t logger = kwiver::vital::get_logger(
    "vital.algorithm" );
  const std::string type_key = name + config_block::block_sep() + "type";

  const std::string type_name = demangle( typeid( INTERFACE ).name() );

  if( config->has_value( type_key ) )
  {
    const std::string iname = config->get_value< std::string >( type_key );
    if( has_algorithm_impl_name< INTERFACE >( iname ) )
    {
      nested_algo = create_algorithm< INTERFACE >( iname );
      nested_algo->set_configuration(
        config->subblock_view( name + config_block::block_sep() + iname )
      );
      LOG_DEBUG(
        logger, "Found Config implementation \"" << iname
                                                 << "\" for \"" << type_name <<
          "\" from key \"" << type_key << "\"" );
    }
    else
    {
      std::stringstream msg;
      msg << "Could not find implementation \"" << iname
          << "\" for \"" << type_name << "\" from key \"" << type_key << "\"";

      // Add line number if known
      std::string file;
      int line( 0 );
      if( config->get_location( type_key, file, line ) )
      {
        msg << " as requested from "
            << file << ":" << line;
      }

      LOG_WARN( logger, msg.str() );
    }
  }
  else
  {
    LOG_WARN(
      logger, "Config item \"" << type_key
                               << "\" not found for \""
                               << type_name << "\"." );
  }
}

//// ----------------------------------------------------------------------------
/// Helper function for properly getting a nested algorithm's configuration
///
/// Adds a configurable algorithm implementation switch for this algorithm.
/// If the variable pointed to by \c nested_algo is a defined sptr to an
/// implementation, its \link kwiver::vital::config_block configuration
/// \endlink
/// parameters are merged with the given
/// \link kwiver::vital::config_block config_block \endlink.
///
/// \tparam          INTERFACE   interface that the nested_algo implements.
/// \param[in]       name        An identifying name for the nested algorithm
/// \param[in,out]   config      The \c config_block instance in which to put
/// the
///                              nested algorithm's configuration.
/// \param[in]       nested_algo The nested algorithm's sptr variable.
template < typename INTERFACE >
void
get_nested_algo_configuration(
  std::string const& name,
  config_block_sptr config,
  std::shared_ptr< INTERFACE > nested_algo )
{
  config_block_description_t type_comment =
    "Algorithm to use for '" + name + "'.\n"
                                      "Must be one of the following options:";

  // Get list of factories for the algo_name
  kwiver::vital::plugin_manager& vpm =
    kwiver::vital::plugin_manager::instance();
  auto fact_list = vpm.get_factories< INTERFACE >();

  for( kwiver::vital::plugin_factory_handle_t a_fact : fact_list )
  {
    std::string reg_name;
    if( !a_fact->get_attribute(
      kwiver::vital::plugin_factory::PLUGIN_NAME,
      reg_name ) )
    {
      continue;
    }

    type_comment += "\n\t- " + reg_name;

    std::string tmp_d;
    if( a_fact->get_attribute(
      kwiver::vital::plugin_factory::
      PLUGIN_DESCRIPTION, tmp_d ) )
    {
      type_comment += " :: " + tmp_d;
    }
  }

  if( nested_algo )
  {
    config_block_sptr cb = nested_algo->get_configuration();
    config->set_value(
      name + config_block::block_sep() + "type",
      nested_algo->impl_name(),
      type_comment );

    config->subblock_view(
      name + config_block::block_sep() +
      nested_algo->impl_name() )
    ->merge_config( cb );
  }
  else if( !config->has_value( name + config_block::block_sep() + "type" ) )
  {
    config->set_value(
      name + config_block::block_sep() + "type",
      "",
      type_comment );
  }
}

/// Helper function for checking that basic nested algorithm configuration is
/// valid
///
/// Check that the expected implementation switch exists and that its value is
/// registered implementation name.
///
/// If the name is valid, we also recursively call check_configuration() on
/// theset implementation. This is done with a fresh create so we don't have
/// to rely on the implementation being defined in the instance this is
/// called from.
///
/// \param     name        An identifying name for the nested algorithm.
/// \param     config  The \c config_block to check.
template < typename INTERFACE >
bool
check_nested_algo_configuration(
  std::string const& name,
  config_block_sptr config )
{
  static kwiver::vital::logger_handle_t logger = kwiver::vital::get_logger(
    "vital.algorithm" );
  const std::string type_key = name + config_block::block_sep() + "type";

  if( !config->has_value( type_key ) )
  {
    LOG_WARN( logger, "Configuration Failure: missing value: " << type_key );
    return false;
  }

  const std::string instance_name =
    config->get_value< std::string >( type_key );
  if( !has_algorithm_impl_name< INTERFACE >( instance_name ) )
  {
    std::stringstream msg;
    msg << "Implementation '" << demangle( instance_name ) <<
      "' for algorithm type "
        << type_key
        <<
      " could not be found.\nMake sure KWIVER_PLUGIN_PATH is set correctly.";

    // Get list of factories for the algo_name
    kwiver::vital::plugin_manager& vpm =
      kwiver::vital::plugin_manager::instance();
    auto fact_list = vpm.get_factories< INTERFACE >();
    bool first { true };

    // Find the one that provides the impl_name
    for( kwiver::vital::plugin_factory_handle_t a_fact : fact_list )
    {
      // Collect a list of all available implementations for this algorithm
      std::string reg_name;
      if( a_fact->get_attribute(
        kwiver::vital::plugin_factory::PLUGIN_NAME,
        reg_name ) )
      {
        if( first )
        {
          first = false;
          msg << "   Available implementations are:";
        }

        msg << "\n      " << reg_name;
      }
    }

    if( first )
    {
      msg << "   There are no implementations available.";
    }

    LOG_WARN( logger, msg.str() );
    return false;
  }

  // recursively check the configuration of the sub-algorithm
  const std::string qualified_name = name + config_block::block_sep() +
                                     instance_name;

  // Need a real algorithm object to check with
  try
  {
    if( !create_algorithm< INTERFACE >( instance_name )->check_configuration(
      config->subblock_view( qualified_name ) ) )
    {
      LOG_WARN(
        logger,  "Configuration Failure Backtrace: "
          << qualified_name );
      return false;
    }
  }
  catch( const kwiver::vital::plugin_factory_not_found& e )
  {
    LOG_WARN( logger, e.what() );
  }
  return true;
}

// ----------------------------------------------------------------------------
// specializations of set/get_config_helper for shared_ptr< algorithm>
// for base implementations see config_helpers.txx

// A helper for populating \p key in \p config based on the  configuration of
// the algorithm given in \p value.
template < typename ValueType,
  typename std::enable_if_t< detail::is_shared_ptr< ValueType >::value,
    bool > = true,
  typename std::enable_if_t< std::is_base_of_v< kwiver::vital::algorithm,
    typename ValueType::element_type >, bool > = true >
void
set_config_helper(
  config_block_sptr config, const std::string& key,
  const ValueType& value,
  [[maybe_unused]] config_block_description_t const& description  =
  config_block_description_t() )
{
  kwiver::vital::get_nested_algo_configuration< typename ValueType::element_type >( key, config, value );
  // We only set a value to assign a description to the key.
  // The value will never be read from the config itself,
  // as this type has a custom specialized accessor
  // that returns a new instance each time.
  config->set_value< ValueType >( key, value, description );
}

// A helper for getting a value from a config block. This specialization is for
// keys that correspond to nested algorithms.
template < typename ValueType,
  typename std::enable_if_t< detail::is_shared_ptr< ValueType >::value,
    bool > = true,
  typename std::enable_if_t< std::is_base_of_v< kwiver::vital::algorithm,
    typename ValueType::element_type >, bool > = true >
ValueType
get_config_helper( config_block_sptr config, config_block_key_t const& key )
{
  ValueType algo;
  kwiver::vital::set_nested_algo_configuration< typename ValueType::element_type >( key, config, algo );
  return algo;
}

// ----------------------------------------------------------------------------
// specializations of set/get_config_helper for std::vector<shared_ptr<
// algorithm>>
// for base implementations see config_helpers.txx

// A helper for populating \p key in \p config based on the  configuration of
// the vector of algorithms given in \p value.
template < typename ValueType,
  // is it a vector<> ?
  typename std::enable_if_t< detail::is_vector< ValueType >::value,
    bool > = true,
  // is it a vector< share_ptr > ?
  typename std::enable_if_t< detail::is_shared_ptr< typename ValueType::value_type >::value, bool > = true,
  // is it a vector< share_ptr< algorithm> > ?
  typename std::enable_if_t< std::is_base_of_v< kwiver::vital::algorithm,
    typename ValueType::value_type::element_type >, bool > = true >
void
set_config_helper(
  config_block_sptr config, const std::string& key,
  const ValueType& value,
  [[maybe_unused]] config_block_description_t const& description  =
  config_block_description_t() )
{
  using AlgoType = typename ValueType::value_type::element_type;

  size_t n = 1;
  auto key_name = [ key ](size_t n){
                    return key + "_" + std::to_string( n );
                  };
  for( auto const& vs : value )
  {
    kwiver::vital::get_nested_algo_configuration< AlgoType >(
      key_name( n ),
      config, vs );
    n++;
  }
  // We only set a value to assign a description to the key.
  // The value will never be read from the config itself,
  // as this type has a custom specialized accessor
  // that returns a new instance each time.
  config->set_value< ValueType >( key, value, description );
}

// A helper for getting a value from a config block. This specialization is for
// keys that correspond to nested algorithms.
template < typename ValueType,
  // is it a vector<> ?
  typename std::enable_if_t< detail::is_vector< ValueType >::value,
    bool > = true,
  // is it a vector< share_ptr > ?
  typename std::enable_if_t< detail::is_shared_ptr< typename ValueType::value_type >::value,
    bool > = true,
  // is it a vector< share_ptr< algorithm> > ?
  typename std::enable_if_t< std::is_base_of_v< kwiver::vital::algorithm,
    typename ValueType::value_type::element_type >, bool > = true >
ValueType
get_config_helper( config_block_sptr config, config_block_key_t const& key )
{
  using AlgoType = typename ValueType::value_type::element_type;

  ValueType algos;

  auto key_name = [ key ](size_t n){
                    return key + "_" + std::to_string( n );
                  };

  size_t n = 1;
  // get algorithm subblock
  vital::config_block_sptr item_config = config->subblock( key_name( n ) );

  while( !item_config->available_values().empty() )
  {
    std::shared_ptr< AlgoType > algo;
    kwiver::vital::set_nested_algo_configuration< AlgoType >(
      key_name( n ),
      config, algo );
    algos.push_back( algo );

    item_config = config->subblock( key_name( ++n ) );
  }
  return algos;
}

} // namespace kwiver::vital

#endif
