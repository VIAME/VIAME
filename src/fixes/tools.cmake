set(__sprokit_tools_fixes_targets)
set(__sprokit_tools_fixes)

option(SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES "Enables fixes for boost::program_options bugs (recommended)" ON)
mark_as_advanced(SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES)
if (SPROKIT_ENABLE_BOOST_PROGRAM_OPTIONS_FIXES)
  if (Boost_VERSION LESS 105000)
    # Override Boost's value_semantic.hpp
    set(reldir "value_name")
    set(target_name "boost-program_options-value_semantic.hpp")
    sprokit_configure_directory(${target_name}
      "${sprokit_fix_source_dir}/${reldir}/boost"
      "${sprokit_fix_binary_dir}/${reldir}/${BOOST_MANGLE_NAMESPACE}")
    set(__sprokit_tools_fixes
      ${__sprokit_tools_fixes}
      "${__sprokit_reldir}")
    set(__sprokit_tools_fixes_targets
      ${__sprokit_tools_fixes_targets}
      ${target_name})
  endif ()

  unset(__sprokit_reldir)
  unset(__sprokit_target_name)
endif ()

function (sprokit_use_tools_fixes)
  foreach (tools_fix ${__sprokit_tools_fixes})
    if (NOT BOOST_MANGLE_NAMESPACE STREQUAL "boost")
      include_directories(BEFORE SYSTEM "${sprokit_fix_source_dir}/${tools_fix}")
    endif ()
    include_directories(BEFORE SYSTEM "${sprokit_fix_binary_dir}/${tools_fix}")
  endforeach ()
endfunction ()

function (sprokit_require_tools_fixes target)
  if (__sprokit_tools_fixes_targets)
    add_dependencies(${target}
      ${__sprokit_tools_fixes_targets})
  endif (__sprokit_tools_fixes_targets)
endfunction ()
