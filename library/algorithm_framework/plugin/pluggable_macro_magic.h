// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef PLUGGABLE_MACRO_MAGIC_H
#define PLUGGABLE_MACRO_MAGIC_H

#include <viame/core_types/cpp_magic.h>
#include <viame/algorithm_framework/plugin/pluggable.h>

#include <utility>

// -----------------
// macro impl config helpers

/*
 * macro for implementations to register a mapping of configurable attributes
 * to config_block properties
 * would need to include:
 *    constructor-param name
 *    description string
 *    default value
 *    constructor param position somehow? maybe?
 *
 * Python equivalents
 * * from_config:
 *     Draw from constructor parameter introspection for
 *
 */

#define TEST_OPT_2( a, b ) a ## b
#define TEST_OPT_3( a, b, c ) a ## b ## c

#define TEST_OPT_ARG( a, b, ... )                                       \
CPP_MAGIC_IF_ELSE( CPP_MAGIC_NOT( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) ) ) \
(                                                                       \
  TEST_OPT_2( a, b ),                                                   \
  TEST_OPT_3( a, b, __VA_ARGS__ )                                       \
)

static int _test_opt_arg{ TEST_OPT_ARG( 1, 2, ) };

// ----------------------------------------------------------------------------
// Helper macros

/**
 * Standard translation of a parameter name to the local member variable
 * the value is stored.
 *
 * This uses the standard prefix "c_" to denote that it is a configured
 * parameter, i.e. one that will be stored in the config_block.
 */
#define CONFIG_VAR_NAME( name ) c_ ## name

/**
 * Conditionally surround the symbol with comments if the second argument is
 * true.
 */
#define MAYBE_COMMENT( symbol, do_comment )       \
CPP_MAGIC_IF_ELSE( CPP_MAGIC_BOOL( do_comment ) ) \
(                                                 \
  /* symbol */,                                   \
  symbol                                          \
)

// ----------------------------------------------------------------------------
// Parameter declaration macros
//
// These macros provide options in declaration, translating those variations
// into a standard structure for the rest of this system to utilize.
//
// Common "parameter" tuple structure format:
//   ( name, type, description_str, default_value )
//
//   Required: name, type, description_str
//   Optional: default_value

/**
 * Declare a parameter with no default value.
 */
#define PARAM( name, type, description_str ) \
( name, type, description_str, )

/**
 * Declare a parameter *with* a default value.
 */
#define PARAM_DEFAULT( name, type, description_str, default ) \
( name, type, description_str, default )

// ----------------------------------------------------------------------------

#define PARAM_VAR_DEF( tuple ) PARAM_VAR_DEF_ tuple
#define PARAM_VAR_DEF_( name, type, description_str, default_value ) \
type CONFIG_VAR_NAME( name );

#define PARAM_PUBLIC_GETTER( tuple ) PARAM_PUBLIC_GETTER_ tuple
#define PARAM_PUBLIC_GETTER_( name, type, description_str, default_value ) \
type const& CPP_MAGIC_CAT( get_, name )( ) const                           \
{                                                                          \
  return this->CONFIG_VAR_NAME( name );                                    \
}

#define PARAM_PUBLIC_SETTER( tuple ) PARAM_PUBLIC_SETTER_ tuple
#define PARAM_PUBLIC_SETTER_( name, type, description_str, default_value ) \
void CPP_MAGIC_CAT( set_, name )( type value )                             \
{                                                                          \
  this->CONFIG_VAR_NAME( name ) = value;                                   \
}

/**
 * Produce a constructor parameter definition and optional default value
 * assignment.
 */
#define PARAM_CONSTRUCTOR_ARGS( tuple ) PARAM_CONSTRUCTOR_ARGS_ tuple
#define PARAM_CONSTRUCTOR_ARGS_( name, type, description_str, default_value ) \
CPP_MAGIC_IF_ELSE( CPP_MAGIC_HAS_ARGS( default_value ) )                      \
(                                                                             \
  type name = ( default_value ),                                              \
  type name = type()                                                          \
)

/**
 * Produce a constructor default assignment, e.g. that placed after the ":" and
 * before the constructor function body.
 */
#define PARAM_CONSTRUCTOR_ASSN( tuple ) PARAM_CONSTRUCTOR_ASSN_ tuple
#define PARAM_CONSTRUCTOR_ASSN_( name, type, description_str, default_value ) \
CONFIG_VAR_NAME( name )( name )

/**
 * Produce an access call to the config_block (assumed variable `cb`) to get a
 * value out, cast to the parameters declared type.
 */
#define PARAM_CONFIG_GET( tuple ) PARAM_CONFIG_GET_ tuple
#define PARAM_CONFIG_GET_( name, type, description_str, default ) \
CPP_MAGIC_IF_ELSE( CPP_MAGIC_HAS_ARGS( default ) )                \
(                                                                 \
  kwiver::vital::get_config_helper< type >( cb, #name, default ), \
  kwiver::vital::get_config_helper< type >( cb, #name )           \
)

#define PARAM_CONFIG_GET_FROM_THIS( tuple ) PARAM_CONFIG_GET_FROM_THIS_ tuple
#define PARAM_CONFIG_GET_FROM_THIS_( name, type, description_str, default ) \
kwiver::vital::set_config_helper< type >(                                   \
  cb, #name,                                                                \
  this->CONFIG_VAR_NAME(                                                    \
  name ), description_str );

/**
 * Produce a set_value call on the config_block (assumed variable `cb`) to set
 * the current class-variable value.
 */
#define PARAM_CONFIG_DEFAULT_SET( tuple ) PARAM_CONFIG_DEFAULT_SET_ tuple
#define PARAM_CONFIG_DEFAULT_SET_( name, type, description_str, \
                                   default )                    \
CPP_MAGIC_IF_ELSE( CPP_MAGIC_HAS_ARGS( default ) )              \
(                                                               \
  kwiver::vital::set_config_helper< type >(                     \
  cb, #name, default,                                           \
  description_str ); ,                                          \
  kwiver::vital::set_config_helper< type >(                     \
  cb, #name, type(),                                            \
  description_str );                                            \
)

// ----------------------------------------------------------------------------

/**
 * The following generation macros hinge on providing an x-macro that expands
 * to enumerate PARAM_* specification tuples as created above.
 */

/**
 * Setup private member variables for the parameter set, as well as public
 * accessor methods that return const& variants of parameter types.
 */
#define PLUGGABLE_VARIABLES( ... )                                   \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )(                   \
private:                                                             \
  CPP_MAGIC_MAP( PARAM_VAR_DEF, CPP_MAGIC_EMPTY, __VA_ARGS__ )       \
public:                                                              \
  CPP_MAGIC_MAP( PARAM_PUBLIC_GETTER, CPP_MAGIC_EMPTY, __VA_ARGS__ ) \
  CPP_MAGIC_MAP( PARAM_PUBLIC_SETTER, CPP_MAGIC_EMPTY, __VA_ARGS__ ) \
)

#define PLUGGABLE_CONSTRUCTOR( class_name, ... )                          \
public:                                                                    \
explicit class_name(                                                      \
  CPP_MAGIC_MAP(                                                          \
  PARAM_CONSTRUCTOR_ARGS, CPP_MAGIC_COMMA,                                \
  __VA_ARGS__ ) )                                                         \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )(                        \
  : CPP_MAGIC_MAP( PARAM_CONSTRUCTOR_ASSN, CPP_MAGIC_COMMA, __VA_ARGS__ ) \
)                                                                         \
{                                                                         \
  this->initialize();                                                     \
}

#define PLUGGABLE_STATIC_FROM_CONFIG( class_name, ... )           \
public:                                                            \
static ::kwiver::vital::pluggable_sptr from_config(               \
  [[maybe_unused]] ::kwiver::vital::config_block_sptr const cb )  \
{                                                                 \
  return std::make_shared< class_name >(                          \
  CPP_MAGIC_MAP( PARAM_CONFIG_GET, CPP_MAGIC_COMMA, __VA_ARGS__ ) \
  );                                                              \
}

#define PLUGGABLE_STATIC_GET_DEFAULT( ... )                               \
public:                                                                    \
static void get_default_config(                                           \
  [[maybe_unused]] ::kwiver::vital::config_block& config )                \
{                                                                         \
  kwiver::vital::config_block_sptr cb =                                   \
    kwiver::vital::config_block::empty_config();                          \
  CPP_MAGIC_MAP( PARAM_CONFIG_DEFAULT_SET, CPP_MAGIC_EMPTY, __VA_ARGS__ ) \
  config.merge_config( cb );                                              \
}

#define PLUGGABLE_GET_CONFIGURATION( ... )                                  \
public:                                                                      \
kwiver::vital::config_block_sptr get_configuration()     const override     \
{                                                                           \
  kwiver::vital::config_block_sptr cb =                                     \
    kwiver::vital::config_block::empty_config();                            \
  CPP_MAGIC_MAP( PARAM_CONFIG_GET_FROM_THIS, CPP_MAGIC_EMPTY, __VA_ARGS__ ) \
  return cb;                                                                \
}                                                                           \


/**
 * Produce a configuration helper for a single parameter
 */
#define PARAM_CONFIG_SET( tuple ) PARAM_CONFIG_SET_ tuple
#define PARAM_CONFIG_SET_( name, type, description_str, default ) \
this->CONFIG_VAR_NAME( name ) =                                   \
  kwiver::vital::get_config_helper< type >( config, #name );      \


/**
 * Define a method for setting an algorithm's configuration
 */
#define PLUGGABLE_SET_CONFIGURATION( class_name, ... )              \
public:                                                              \
void set_configuration(                                             \
  ::kwiver::vital::config_block_sptr in_config )  override          \
{                                                                   \
  kwiver::vital::config_block_sptr config =                         \
    kwiver::vital::config_block::empty_config();                    \
  class_name::get_default_config( *config );                        \
  config->merge_config( in_config );                                \
  CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )(                \
    CPP_MAGIC_MAP( PARAM_CONFIG_SET, CPP_MAGIC_EMPTY, __VA_ARGS__ ) \
  )                                                                 \
  this->set_configuration_internal( in_config );                    \
}                                                                   \

// ----------------------------------------------------------------------------

/**
 * Define necessary static methods for pluggable interfaces.
 *
 * \param interface_name The name of the interface class, or other like string
 * that will be used as the string name for this interface.
 */
#define PLUGGABLE_INTERFACE( name ) \
public:                              \
static std::string interface_name() \
{                                   \
  return #name;                     \
}                                   \
virtual ~name() = default;

/**
 * Variant for classes that define their own destructor.
 */
#define PLUGGABLE_INTERFACE_NO_DESTR( name ) \
public:                                       \
static std::string interface_name()          \
{                                            \
  return #name;                              \
}

/**
 * Basic implementation class helper macro for when you want to author your
 * own from_config and get_default_config static methods.
 */
#define PLUGGABLE_IMPL_BASIC( class_name, description ) \
public:                                                  \
static std::string plugin_name()                        \
{                                                       \
  return #class_name;                                   \
}                                                       \
static std::string plugin_description()                 \
{                                                       \
  return description;                                   \
}

/**
 * As PLUGGABLE_IMPL_BASIC, but the name the plugin registers under is given
 * explicitly instead of being taken from the class name. Needed wherever two
 * implementations of the same interface share a class name (for example a
 * plain and an OpenCV-backed variant), or where the registered name has to
 * stay stable across a class rename.
 */
#define PLUGGABLE_IMPL_BASIC_NAMED( class_name, plugin_name_str, description ) \
public:                                                                         \
static std::string plugin_name()                                               \
{                                                                              \
  return plugin_name_str;                                                      \
}                                                                              \
static std::string plugin_description()                                        \
{                                                                              \
  return description;                                                          \
}

/**
 * As PLUGGABLE_IMPL, with an explicit registered plugin name.
 */
#define PLUGGABLE_IMPL_NAMED( class_name, plugin_name_str, description, ... ) \
PLUGGABLE_VARIABLES( __VA_ARGS__ )                                            \
PLUGGABLE_CONSTRUCTOR( class_name, __VA_ARGS__ )                              \
PLUGGABLE_IMPL_BASIC_NAMED( class_name, plugin_name_str, description )        \
PLUGGABLE_STATIC_FROM_CONFIG( class_name, __VA_ARGS__ )                       \
PLUGGABLE_STATIC_GET_DEFAULT( __VA_ARGS__ )                                   \
PLUGGABLE_SET_CONFIGURATION( class_name, __VA_ARGS__ )                        \
PLUGGABLE_GET_CONFIGURATION( __VA_ARGS__ )                                    \


/**
 * All together now: TODO detail composition
 */
#define PLUGGABLE_IMPL( class_name, description, ... )  \
/** @brief doxy_comment */                              \
PLUGGABLE_VARIABLES( __VA_ARGS__ )                      \
PLUGGABLE_CONSTRUCTOR( class_name, __VA_ARGS__ )        \
PLUGGABLE_IMPL_BASIC( class_name, description )         \
PLUGGABLE_STATIC_FROM_CONFIG( class_name, __VA_ARGS__ ) \
PLUGGABLE_STATIC_GET_DEFAULT( __VA_ARGS__ )             \
PLUGGABLE_SET_CONFIGURATION( class_name, __VA_ARGS__ )  \
PLUGGABLE_GET_CONFIGURATION( __VA_ARGS__ )              \


// ----------------------------------------------------------------------------
// utilties for PIMPL
// TODO document why nwe need them
namespace kwiver::vital::detail {

template < typename T >
void
KwiverDefaultDeleter( T* p )
{
  delete p;
}

template < typename T >
void
KwiverEmptyDeleter( T* p )
{
  ( void ) ( p );
}

} // namespace kwiver::vital::detail

#define KWIVER_UNIQUE_PTR( type, name ) \
std::unique_ptr< type,                  \
  decltype( &kwiver::                   \
            vital::                     \
            detail::                    \
            KwiverEmptyDeleter          \
            < type > ) >                \
name = { nullptr, kwiver::vital::detail::KwiverEmptyDeleter< type > }
#define KWIVER_INITIALIZE_UNIQUE_PTR( type,                                      \
                                      name ) this->name = std::unique_ptr< type, \
  decltype( &kwiver::vital::detail::KwiverDefaultDeleter< type > ) >(            \
  new type(                                                                      \
  *this ), kwiver::vital::detail::KwiverDefaultDeleter< type > )
// ----------------------------------------------------------------------------

#define KWIVER_STRINGIFY( x ) #x

#endif // PLUGGABLE_MACRO_MAGIC_H
