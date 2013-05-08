set(__sprokit_python_fixes_targets)
set(__sprokit_python_fixes)

option(SPROKIT_ENABLE_BOOST_PYTHON_FIXES "Enables fixes for Boost.Python bugs (recommended)" ON)
mark_as_advanced(SPROKIT_ENABLE_BOOST_PYTHON_FIXES)
if (SPROKIT_ENABLE_BOOST_PYTHON_FIXES)
  # Not fixed yet.
  #if (Boost_VERSION LESS 10XX00)
    # Override Boost's invoke.hpp
    set(__sprokit_reldir "fixes/threading")
    set(__sprokit_target_name "boost-python-invoke.hpp")
    sprokit_configure_directory(${__sprokit_target_name}
      "${sprokit_fix_source_dir}/${__sprokit_reldir}/boost"
      "${sprokit_fix_binary_dir}/${__sprokit_reldir}/${BOOST_MANGLE_NAMESPACE}")
    set(__sprokit_python_fixes
      ${__sprokit_python_fixes}
      "${__sprokit_reldir}")
    set(__sprokit_python_fixes_targets
      ${__sprokit_python_fixes_targets}
      ${__sprokit_target_name})
  #endif ()
  # Not fixed yet.
  #if (Boost_VERSION LESS 10XX00)
    # Override Boost's override.hpp
    set(__sprokit_reldir "fixes/exceptions_in_override")
    set(__sprokit_target_name "boost-python-override.hpp")
    sprokit_configure_directory(${__sprokit_target_name}
      "${sprokit_fix_source_dir}/${__sprokit_reldir}/boost"
      "${sprokit_fix_binary_dir}/${__sprokit_reldir}/${BOOST_MANGLE_NAMESPACE}")
    set(__sprokit_python_fixes
      ${__sprokit_python_fixes}
      "${__sprokit_reldir}")
    set(__sprokit_python_fixes_targets
      ${__sprokit_python_fixes_targets}
      ${__sprokit_target_name})
  #endif ()

  unset(__sprokit_reldir)
  unset(__sprokit_target_name)
endif ()

function (sprokit_use_python_fixes)
  foreach (python_fix ${__sprokit_python_fixes})
    if (NOT BOOST_MANGLE_NAMESPACE STREQUAL "boost")
      include_directories(BEFORE SYSTEM "${sprokit_fix_source_dir}/${python_fix}")
    endif ()
    include_directories(BEFORE SYSTEM "${sprokit_fix_binary_dir}/${python_fix}")
  endforeach ()
endfunction ()

function (sprokit_require_python_fixes target)
  if (__sprokit_python_fixes_targets)
    add_dependencies(${target}
      ${__sprokit_python_fixes_targets})
  endif (__sprokit_python_fixes_targets)
endfunction ()
