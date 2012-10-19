set(hooks_directory
  "${vistk_source_dir}/.git/hooks")

add_custom_command(
  OUTPUT  "${hooks_directory}/.git"
  COMMAND "${GIT_EXECUTABLE}"
          init
  COMMAND "${GIT_EXECUTABLE}"
          remote add origin
          ../..
  WORKING_DIRECTORY
          "${hooks_directory}"
  COMMENT "Initializing the hooks repository")
add_custom_target(hooks-init
  DEPENDS
    "${hooks_directory}/.git")

add_custom_target(hooks-update ALL
  DEPENDS
    hooks-init)
add_custom_command(
  TARGET  hooks-update
  COMMAND "${GIT_EXECUTABLE}"
          fetch
          --quiet
          origin
          remotes/origin/hooks
  COMMAND "${GIT_EXECUTABLE}"
          merge
          --quiet
          FETCH_HEAD
  WORKING_DIRECTORY
          "${hooks_directory}"
  COMMENT "Updating hooks repository")
