// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Macros for pragmatically creating pipeline descriptions.
 *
 * These macros are used to dynamically create a pipeline description
 * at run time.
 *
 * \code
  // Use SPROKIT macros to create pipeline description
  std::stringstream pipeline_desc;
  pipeline_desc

//   Macro form of pipeline                           resulting pipeline text
// -------------------------------------------------------------------------------
<< SPROKIT_CONFIG_BLOCK( "multiplier )             // config multiplier
<< SPROKIT_CONFIG( "start1", "10" )                //  start1 = 10
<< SPROKIT_CONFIG( "end1", "20" )                  //  end1   = 20
<< SPROKIT_CONFIG( "start2", "10" )                //  start2 = 10
<< SPROKIT_CONFIG( "end2", "30" )                  //  end2   = 30
<< SPROKIT_CONFIG( "output", "products.txt" )      //  output = products.txt

<< SPROKIT_PROCESS( "gen_numbers1", "numbers" ) // process gen_numbers1 :: numbers
<< SPROKIT_CONFIG_FULL( "start", "ro", "$CONFIG{multiplier:start1"} ) //  start[ro] =  $CONFIG{multiplier:start1}
<< SPROKIT_CONFIG_FULL( "end", "ro", "$CONFIG{multiplier:end1}" )      //  end[ro]  = $CONFIG{multiplier:end1}

<< SPROKIT_PROCESS( "gen_numbers2", "numbers" )  // process gen_numbers2 :: numbers
<< SPROKIT_CONFIG_FULL( "start", "ro", "$CONFIG{multiplier:start2}" ) //  start[ro]  = $CONFIG{multiplier:start2}
<< SPROKIT_CONFIG_FULL( "end", "ro",  "$CONFIGP{multiplier:end2}" )   //  end[ro]    = $CONFIG{multiplier:end2)

<< SPROKIT_PROCESS( "multiply", "multiplication")   //  process multiply :: multiplication

<< SPROKIT_PROCESS( "print", "print_number" )       //process print :: print_number
<< SPROKIT_CONFIG_FULL( "output", "ro", $CONFIG{multiplier:output}" ) //  output[ro]   = $CONFIG{multiplier:output}

<< SPROKIT_CONNECT( "gen_numbers1", "number",    "multiply", "factor1" ) //connect from gen_numbers1.number
                                                                         //        to   multiply.factor1
<< SPROKIT_CONNECT( "gen_numbers2", "number",    "multiply", "factor2" ) // connect from gen_numbers2.number
                                                                         //        to   multiply.factor2
<< SPROKIT_CONNECT( "multiply", "product",     "print", "number" )       // connect from multiply.product
                                                                         //        to   print.number
    ;
 * \endcode
 */

#ifndef SPROKIT_TOOLS_LITERAL_PIPELINE_H
#define SPROKIT_TOOLS_LITERAL_PIPELINE_H

/**
 * @brief Define a process in the pipeline.
 *
 * This macro defines a process that is to be part of the pipeline. If
 * this process needs any config parameters, they must follow the
 * process definition.
 *
 * @param type Process type name
 * @param name Process instance name in this pipeline.
 */
#define SPROKIT_PROCESS(type, name) \
  "process " << name << "  :: " << type << "\n"

#define SPROKIT_FLAGS_WRAP(flags) \
  "[" << flags << "]"

#define SPROKIT_CONFIG_RAW(key, flags, value) \
  key << flags << " = " << value << "\n"

/**
 * @brief Fully specify a config item entry.
 *
 * This macro generates a fully specified configuration item.
 *
 * @param key Name of configuration item.
 * @param flags Flags for this configuration item.
 * @param value Value of this configuration item.
 */
#define SPROKIT_CONFIG_FULL(key, flags, value) \
  SPROKIT_CONFIG_RAW(key, SPROKIT_FLAGS_WRAP(flags), value)

/**
 * @brief Specify config item with flags.
 *
 * This macro generates a configuration item with flags.
 *
 * @param key Name of configuration item.
 * @param flags Flags for this configuration item.
 * @param value Value of this configuration item.
 */
#define SPROKIT_CONFIG_FLAGS(key, flags, value) \
  SPROKIT_CONFIG_RAW(key, SPROKIT_FLAGS_WRAP(flags), value)

/**
 * @brief Specify a configuration item.
 *
 * @param key Configuration item name or key.
 * @param value Value for this configuration item.
 */
#define SPROKIT_CONFIG(key, value) \
  SPROKIT_CONFIG_RAW(key, "", value)

/**
 * @brief Start a named configuration block.
 *
 * This macro inserts a "config <name>" line in the config starting a
 * named config block. Note that this is different than a nested
 * config block.
 *
 * @param name Name of the config block.
 */
#define SPROKIT_CONFIG_BLOCK(name) \
  "config " << name << "\n"

/**
 * @brief Start a nested config block.
 *
 * This macro inserts a BLOCK <name> line in the configuration,
 * starting a nested config block. Note that this is different than
 * starting a named config block.
 *
 * @param name - Name of the nested block.
 */
#define SPROKIT_CONFIG_NESTED_BLOCK(name) \
  "block " << name << "\n"

/**
 * @brief End a nested config block.
 *
 * This macro inserts an "ENDBLOCK" line in the config terminating the
 * current nested config block level. This does not have any effect on
 * the named config block.
 */
#define SPROKIT_CONFIG_NESTED_BLOCK_END() \
  "endblock\n"

/**
 * @brief Define a connection between two ports.
 *
 * This macro generates a single connection between two processes.
 *
 * @param up_name Upstream process name.
 * @param up_port Upstream port name.
 * @param down_name Downstream process name.
 * @param down_port Downstream port name.
 */
#define SPROKIT_CONNECT(up_name, up_port, down_name, down_port) \
  "connect from " << up_name << "." << up_port                  \
  << "  to  " << down_name << "." << down_port << "\n"

#endif // SPROKIT_TOOLS_LITERAL_PIPELINE_H
