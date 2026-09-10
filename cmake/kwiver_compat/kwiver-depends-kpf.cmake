# external dependency for kpf i/o

kwiver_package_option(KPF
  DESCRIPTION "Enable kpf i/o for vital types"
  FLETCH_NAME YAMLCPP
)
mark_as_advanced( KWIVER_ENABLE_KPF )

if( kwiver_enabled_kpf )
  add_definitions( -DKWIVER_ENABLE_KPF )
endif()
