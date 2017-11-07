# external dependency for kpf i/o

option( KWIVER_ENABLE_KPF
  "Enable kpf i/o for vital types"
  ${fletch_ENABLED_YAMLCPP}
)
# Mark this as advanced 
mark_as_advanced( KWIVER_ENABLE_KPF )
