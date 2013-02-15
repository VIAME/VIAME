execute_process(
  COMMAND           date
                    +%Y%m%d
  WORKING_DIRECTORY "${vistk_source_dir}"
  RESULT_VARIABLE   date_return
  OUTPUT_VARIABLE   vistk_date)
execute_process(
  COMMAND           date
                    +%H%M%S
  WORKING_DIRECTORY "${vistk_source_dir}"
  RESULT_VARIABLE   time_return
  OUTPUT_VARIABLE   vistk_time)
execute_process(
  COMMAND           "${GIT_EXECUTABLE}"
                    rev-parse
                    --short
                    HEAD
  WORKING_DIRECTORY "${vistk_source_dir}"
  RESULT_VARIABLE   git_return
  OUTPUT_VARIABLE   vistk_git_hash_short)
execute_process(
  COMMAND           "${GIT_EXECUTABLE}"
                    diff
                    --no-ext-diff
                    --quiet
                    --exit-code
  WORKING_DIRECTORY "${vistk_source_dir}"
  RESULT_VARIABLE   git_return)

string(STRIP "${vistk_date}" vistk_date)
string(STRIP "${vistk_time}" vistk_time)
string(STRIP "${vistk_git_hash_short}" vistk_git_hash_short)

set(snap_suffix ".${vistk_date}git${vistk_git_hash_short}")
set(dirty_suffix ".dirty${vistk_time}")

execute_process(
  COMMAND           "${GIT_EXECUTABLE}"
                    archive
                    --format=tar
                    --prefix=vistk-${vistk_version}/
                    HEAD
  COMMAND           bzip2
  OUTPUT_FILE       "${vistk_binary_dir}/vistk-${vistk_version}${snap_suffix}.tar.bz2"
  WORKING_DIRECTORY "${vistk_source_dir}")
if (git_return)
  execute_process(
    COMMAND           "${GIT_EXECUTABLE}"
                      diff
                      HEAD
    OUTPUT_FILE       "${vistk_binary_dir}/vistk-${vistk_version}${snap_suffix}${dirty_suffix}.patch"
    WORKING_DIRECTORY "${vistk_source_dir}")
endif ()
