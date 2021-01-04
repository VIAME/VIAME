// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_IMPL_H
#define SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_IMPL_H

#include "function_process.h"

#include <boost/tuple/tuple.hpp>

/**
 * \file function_process_impl.h
 *
 * \brief Macros for implementing a process which wraps a function.
 */

/**
 * \def IPORT_VAR
 *
 * \brief The name of the variable which holds the name of an input port.
 *
 * \param port The name of the port.
 */
#define IPORT_VAR(port) port_input_##port
/**
 * \def OPORT_VAR
 *
 * \brief The name of the variable which holds the name of an output port.
 *
 * \param port The name of the port.
 */
#define OPORT_VAR(port) port_output_##port
/**
 * \def LOCAL_IPORT_VAR
 *
 * \brief The name of a local variable which holds the data from an input port.
 *
 * \param port The name of the port.
 */
#define LOCAL_IPORT_VAR(port) local_input_##port
/**
 * \def LOCAL_OPORT_VAR
 *
 * \brief The name of a local variable which holds the data to be sent to an output port.
 *
 * \param port The name of the port.
 */
#define LOCAL_OPORT_VAR(port) local_output_##port

/**
 * \def CONF_ARG_VAR
 *
 * \brief The argument variable for a configuration value.
 *
 * \param key The name of the configuration key.
 */
#define CONF_ARG_VAR(key) conf_arg_##key
/**
 * \def CONF_VAR
 *
 * \brief The variable for a configuration key.
 *
 * \param key The name of the configuration key.
 */
#define CONF_VAR(key) conf_key_##key
/**
 * \def CONF_VALUE_VAR
 *
 * \brief The variable for the value of a configuration key.
 *
 * \param key The name of the configuration key.
 */
#define CONF_VALUE_VAR(key) conf_value_##key
/**
 * \def CONF_DEF_VAR
 *
 * \brief The default value for a configuration key.
 *
 * \param key The name of the configuration key.
 */
#define CONF_DEF_VAR(key) conf_default_##key
/**
 * \def LOCAL_CONF_VAR
 *
 * \brief The name of a local variable holding a configuration value.
 *
 * \param key The name of the configuration key.
 */
#define LOCAL_CONF_VAR(key) local_conf_##key

/**
 * \def DECLARE_IPORT_VARS
 *
 * \brief Declares static variables for the names of input ports.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define DECLARE_IPORT_VARS(cls, name, type, port_type, flags, desc) \
  static port_t const IPORT_VAR(name)
/**
 * \def DEFINE_IPORT_VARS
 *
 * \brief Defines static variables for the names of input ports.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define DEFINE_IPORT_VARS(cls, name, type, port_type, flags, desc) \
  sprokit::process::port_t const CLASS_NAME(cls)::priv::IPORT_VAR(name) = sprokit::process::port_t(#name)

/**
 * \def DECLARE_OPORT_VARS
 *
 * \brief Declares static variables for the names of output ports.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define DECLARE_OPORT_VARS(cls, name, type, port_type, flags, desc) \
  static port_t const OPORT_VAR(name)
/**
 * \def DEFINE_OPORT_VARS
 *
 * \brief Defines static variables for the names of output ports.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define DEFINE_OPORT_VARS(cls, name, type, port_type, flags, desc) \
  sprokit::process::port_t const CLASS_NAME(cls)::priv::OPORT_VAR(name) = sprokit::process::port_t(#name)

/**
 * \def DECLARE_CONF_VARS
 *
 * \brief Declares variables for the name, default, and storage of a configuration.
 *
 * \param cls The C++ class the configuration is for.
 * \param name The name of configuration key.
 * \param type The C++ type for the value retrieved from the configuration.
 * \param default The default value of the configuration.
 * \param desc A description of the configuration.
 */
#define DECLARE_CONF_VARS(cls, name, type, default, desc) \
  static kwiver::vital::config_block_key_t const CONF_VAR(name);       \
  static kwiver::vital::config_block_value_t const CONF_DEF_VAR(name); \
  type const CONF_VALUE_VAR(name)
/**
 * \def DEFINE_CONF_VARS
 *
 * \brief Defines static variables for the names of configuration keys and defaults.
 *
 * \param cls The C++ class the configuration is for.
 * \param name The name of configuration key.
 * \param type The C++ type for the value retrieved from the configuration.
 * \param default The default value of the configuration.
 * \param desc A description of the configuration.
 */
#define DEFINE_CONF_VARS(cls, name, type, default, desc)                                          \
  kwiver::vital::config_block_key_t const CLASS_NAME(cls)::priv::CONF_VAR(name) = kwiver::vital::config_block_key_t(#name); \
  kwiver::vital::config_block_value_t const CLASS_NAME(cls)::priv::CONF_DEF_VAR(name) = kwiver::vital::config_block_value_t(default)

/**
 * \def CONFIG_DECLARE_ARGS
 *
 * \brief Expands to an argument to a function taking the configuration.
 *
 * \param cls The C++ class the configuration is for.
 * \param name The name of configuration key.
 * \param type The C++ type for the value retrieved from the configuration.
 * \param default The default value of the configuration.
 * \param desc A description of the configuration.
 */
#define CONFIG_DECLARE_ARGS(cls, name, type, default, desc) \
  type const& CONF_ARG_VAR(name)

/**
 * \def DECLARE_IPORT
 *
 * \brief Declares an input port for the process.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define DECLARE_IPORT(cls, name, type, port_type, flags, desc) \
  declare_input_port(                                          \
    priv::IPORT_VAR(name),                                     \
    port_type,                                                 \
    flags,                                                     \
    port_description_t(desc))
/**
 * \def DECLARE_OPORT
 *
 * \brief Declares an output port for the process.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define DECLARE_OPORT(cls, name, type, port_type, flags, desc) \
  declare_output_port(                                         \
    priv::OPORT_VAR(name),                                     \
    port_type,                                                 \
    flags,                                                     \
    port_description_t(desc))
/**
 * \def DECLARE_CONFIG
 *
 * \brief Declares a configuration value for the process.
 *
 * \param cls The C++ class the configuration is for.
 * \param name The name of configuration key.
 * \param type The C++ type for the value retrieved from the configuration.
 * \param default The default value of the configuration.
 * \param desc A description of the configuration.
 */
#define DECLARE_CONFIG(cls, name, type, default, desc) \
  declare_configuration_key(                           \
    priv::CONF_VAR(name),                              \
    priv::CONF_DEF_VAR(name),                          \
    kwiver::vital::config_block_description_t(desc))

/**
 * \def GRAB_CONFIG_VALUE
 *
 * \brief Grabs the configuration value for the process.
 *
 * \param cls The C++ class the configuration is for.
 * \param name The name of configuration key.
 * \param type The C++ type for the value retrieved from the configuration.
 * \param default The default value of the configuration.
 * \param desc A description of the configuration.
 */
#define GRAB_CONFIG_VALUE(cls, name, type, default, desc) \
  type const LOCAL_CONF_VAR(name) = config_value<type>(priv::CONF_VAR(name))

/**
 * \def CONF_ARGS
 *
 * \brief Passes configuration values as arguments.
 *
 * \param cls The C++ class the configuration is for.
 * \param name The name of configuration key.
 * \param type The C++ type for the value retrieved from the configuration.
 * \param default The default value of the configuration.
 * \param desc A description of the configuration.
 */
#define CONF_ARGS(cls, name, type, default, desc) \
  LOCAL_CONF_VAR(name)

/**
 * \def CONF_INIT_PRIV
 *
 * \brief The initializer list for configuration values.
 *
 * \param cls The C++ class the configuration is for.
 * \param name The name of configuration key.
 * \param type The C++ type for the value retrieved from the configuration.
 * \param default The default value of the configuration.
 * \param desc A description of the configuration.
 */
#define CONF_INIT_PRIV(cls, name, type, default, desc) \
  CONF_VALUE_VAR(name)(CONF_ARG_VAR(name))

/**
 * \def GRAB_FROM_IPORT
 *
 * \brief Grabs a datum from an input port.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define GRAB_FROM_IPORT(cls, name, type, port_type, flags, desc) \
  type const LOCAL_IPORT_VAR(name) = grab_from_port_as<type>(priv::IPORT_VAR(name))

/**
 * \def DECLARE_RESULT_VARS
 *
 * \brief Declares result variables from the function for the output port.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define DECLARE_RESULT_VARS(cls, name, type, port_type, flags, desc) \
  type LOCAL_OPORT_VAR(name);

/**
 * \def IPORT_ARGS
 *
 * \brief Passes the input data as arguments.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define IPORT_ARGS(cls, name, type, port_type, flags, desc) \
  LOCAL_IPORT_VAR(name)

/**
 * \def RESULT
 *
 * \brief Expands to the fill the output variables from the function.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define RESULT(cls, name, type, port_type, flags, desc) \
  LOCAL_OPORT_VAR(name)

/**
 * \def PUSH_TO_OPORT
 *
 * \brief Pushes the result to the output port.
 *
 * \param cls The C++ class the port is for.
 * \param name The name of the port.
 * \param type The C++ type for the value retrieved from the port.
 * \param port_type The type of the port.
 * \param flags Flags for the port.
 * \param desc A description of the port.
 */
#define PUSH_TO_OPORT(cls, name, type, port_type, flags, desc) \
  push_to_port_as<type>(priv::OPORT_VAR(name), LOCAL_OPORT_VAR(name))

/**
 * \def COMMA
 *
 * \brief A macro which expands to a comma.
 *
 * It is used to avoid confusing the preprocessor parser.
 */
#define COMMA ,
/**
 * \def SEMICOLON
 *
 * \brief A macro which expands to a semicolon.
 *
 * Not necessary, <em>per se</em>, but is used more to match the \ref COMMA
 * define.
 */
#define SEMICOLON ;
/**
 * \def COLON
 *
 * \brief A macro which expands to a colon.
 *
 * Not necessary, <em>per se</em>, but is used more to match the other symbol
 * defines.
 */
#define COLON :

/**
 * \def BEG
 *
 * \brief Expands into the starting symbol for the context of an expansion.
 *
 * \param ctx The context.
 */
#define BEG(ctx) BEG_##ctx
/**
 * \def SEP
 *
 * \brief Expands into the separator symbol for the context of an expansion.
 *
 * \param ctx The context.
 */
#define SEP(ctx) SEP_##ctx
/**
 * \def END
 *
 * \brief Expands into the terminal symbol for the context of an expansion.
 *
 * \param ctx The context.
 */
#define END(ctx) END_##ctx

/// The starting symbol when expansion becomes lines.
#define BEG_LINES
///The separator symbol when expansion becomes lines.
#define SEP_LINES SEMICOLON
///The terminal symbol when expansion becomes lines.
#define END_LINES SEMICOLON

///The terminal symbol when expansion becomes an argument list.
#define BEG_ARGS
///The terminal symbol when expansion becomes an argument list.
#define SEP_ARGS COMMA
///The terminal symbol when expansion becomes an argument list.
#define END_ARGS

///The terminal symbol when expansion becomes an initializer list.
#define BEG_INIT COLON
///The terminal symbol when expansion becomes an initializer list.
#define SEP_INIT COMMA
///The terminal symbol when expansion becomes an initializer list.
#define END_INIT

/**
 * \def IMPLEMENT_FUNCTION_PROCESS
 *
 * \brief Expands to code which implements a process which wraps a function.
 *
 * \note No trailing semicolon is needed.
 *
 * This macro helps to minimize writing the code to get a function into the
 * pipeline. The other macros in this file are used to support this one and
 * should not be used directly.
 *
 * \note The current contract is that data from input ports are passed to \p
 * func in order, each one as a separate parameter and in order, and that its
 * return value is a tuple of the output ports in order.
 *
 * Configuration values can be requested but are currently not available to the
 * function.
 *
 * \note There must be at least one configuration variable currently. This is
 * due to the way that the macros expand and C++ syntax.
 *
 * The currently available flags available for ports include:
 *
 * <dl>
 * \term{port_flags_t()}
 *   \termdef{No flags are set for the port.}
 * \term{required}
 *   \termdef{The port is marked as being required.}
 * </dl>
 *
 * It is highly recommended that all ports are marked as \flag{required}.
 *
 * \note The types of ports are recommended to have cheap copy constructors
 * since that is what is called when grabbing from ports and pushing them.
 *
 * The contract for the \p conf, \p iports, and \p oports parameters is that it
 * must be the name of a macro which takes two arguments:
 *
 * \arg call This is the name of a macro which takes parameters which describe
 *           configuration variables or ports, described below.
 * \arg sep The symbol that is needed between all calls to \p call.
 *
 * \section conf_macro Configuration macro call arguments
 *
 * \arg cls The C++ class the configuration is for.
 * \arg name The name of the configuration key.
 * \arg type The C++ type for the value retrieved from the configuration.
 * \arg default The default value for the configuration.
 * \arg desc A description of configuration key.
 *
 * \section port_macro Port macro call arguments
 *
 * \arg cls The C++ class the port is for.
 * \arg name The name of the port.
 * \arg type The C++ type for the value retrieved from the port.
 * \arg port_type The type of the port.
 * \arg flags Flags for the port.
 * \arg desc A description of the port.
 *
 * \note If there are no input or output ports, then there may be warnings about
 * spurious semicolons.
 *
 * \todo Set up more flags for ports.
 * \todo Need to support no configration variables.
 * \todo How to use configuration variables?
 *
 * \param name The name of the class to implement.
 * \param func The function to call each step.
 * \param conf The macro which expands to information for configuration values.
 * \param iports The macro which expands to information for input ports.
 * \param oports The macro which expands to information for output ports.
 */
#define IMPLEMENT_FUNCTION_PROCESS(name, func, conf, iports, oports)  \
class CLASS_NAME(name)::priv                                          \
{                                                                     \
  public:                                                             \
    priv(conf(CONFIG_DECLARE_ARGS, ARGS));                            \
    ~priv();                                                          \
                                                                      \
    conf(DECLARE_CONF_VARS, LINES)                                    \
    iports(DECLARE_IPORT_VARS, LINES)                                 \
    oports(DECLARE_OPORT_VARS, LINES)                                 \
};                                                                    \
                                                                      \
conf(DEFINE_CONF_VARS, LINES)                                         \
iports(DEFINE_IPORT_VARS, LINES)                                      \
oports(DEFINE_OPORT_VARS, LINES)                                      \
                                                                      \
CLASS_NAME(name)                                                        \
::CLASS_NAME(name)(kwiver::vital::config_block_sptr const& config)      \
                  : sprokit::process(config)                            \
{                                                                     \
  port_flags_t required;                                              \
                                                                      \
  required.insert(flag_required);                                     \
                                                                      \
  conf(DECLARE_CONFIG, LINES)                                         \
  iports(DECLARE_IPORT, LINES)                                        \
  oports(DECLARE_OPORT, LINES)                                        \
}                                                                     \
                                                                      \
CLASS_NAME(name)                                                      \
::CLASS_DTOR(name)()                                                  \
{                                                                     \
}                                                                     \
                                                                      \
void                                                                  \
CLASS_NAME(name)                                                      \
::_configure()                                                        \
{                                                                     \
  conf(GRAB_CONFIG_VALUE, LINES)                                      \
                                                                      \
  d.reset(new priv(conf(CONF_ARGS, ARGS)));                           \
                                                                      \
  process::_configure();                                              \
}                                                                     \
                                                                      \
void                                                                  \
CLASS_NAME(name)                                                      \
::_step()                                                             \
{                                                                     \
  iports(GRAB_FROM_IPORT, LINES)                                      \
                                                                      \
  oports(DECLARE_RESULT_VARS, LINES)                                  \
                                                                      \
  boost::tie(oports(RESULT, ARGS)) = func(iports(IPORT_ARGS, ARGS));  \
                                                                      \
  oports(PUSH_TO_OPORT, LINES)                                        \
                                                                      \
  process::_step();                                                   \
}                                                                     \
                                                                      \
CLASS_NAME(name)::priv                                                \
::priv(conf(CONFIG_DECLARE_ARGS, ARGS))                               \
  conf(CONF_INIT_PRIV, INIT)                                          \
{                                                                     \
}                                                                     \
                                                                      \
CLASS_NAME(name)::priv                                                \
::~priv()                                                             \
{                                                                     \
}

#endif // SPROKIT_PROCESSES_HELPER_FUNCTIONS_FUNCTION_PROCESS_IMPL_H
