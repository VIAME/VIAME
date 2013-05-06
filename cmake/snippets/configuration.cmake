# Add defines for code that care about configuration types.

foreach (config ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER "${config}" upper_config)

  set(config_defines
    "SPROKIT_CONFIGURATION=\"${config}\""
    "SPROKIT_CONFIGURATION_L=L\"${config}\"")

  set_directory_properties(
    PROPERTIES
      COMPILE_DEFINITIONS_${upper_config} "${config_defines}")
endforeach ()
