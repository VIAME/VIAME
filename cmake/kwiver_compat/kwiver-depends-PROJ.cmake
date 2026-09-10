# Optionally find and configure PROJ dependency

kwiver_package_option(PROJ
  DESCRIPTION "Enable PROJ dependent code and plugins (Arrows)"
)

if( kwiver_enabled_proj )
  find_package( PROJ REQUIRED )
endif()
