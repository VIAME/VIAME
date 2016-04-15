execute_process(
  COMMAND           date
                    +%Y%m%d
  WORKING_DIRECTORY "${sprokit_source_dir}"
  RESULT_VARIABLE   date_return
  OUTPUT_VARIABLE   sprokit_date)
execute_process(
  COMMAND           date
                    +%H%M%S
  WORKING_DIRECTORY "${sprokit_source_dir}"
  RESULT_VARIABLE   time_return
  OUTPUT_VARIABLE   sprokit_time)
execute_process(
  COMMAND           "${GIT_EXECUTABLE}"
                    rev-parse
                    --short
                    HEAD
  WORKING_DIRECTORY "${sprokit_source_dir}"
  RESULT_VARIABLE   git_return
  OUTPUT_VARIABLE   sprokit_git_hash_short)
execute_process(
  COMMAND           "${GIT_EXECUTABLE}"
                    diff
                    --no-ext-diff
                    --quiet
                    --exit-code
  WORKING_DIRECTORY "${sprokit_source_dir}"
  RESULT_VARIABLE   git_return)

string(STRIP "${sprokit_date}" sprokit_date)
string(STRIP "${sprokit_time}" sprokit_time)
string(STRIP "${sprokit_git_hash_short}" sprokit_git_hash_short)

set(snap_suffix ".${sprokit_date}git${sprokit_git_hash_short}")
set(dirty_suffix ".dirty${sprokit_time}")

execute_process(
  COMMAND           "${GIT_EXECUTABLE}"
                    archive
                    --format=tar
                    --prefix=sprokit-${sprokit_version}/
                    HEAD
  COMMAND           bzip2
  OUTPUT_FILE       "${sprokit_binary_dir}/sprokit-${sprokit_version}${snap_suffix}.tar.bz2"
  WORKING_DIRECTORY "${sprokit_source_dir}")
if (git_return)
  execute_process(
    COMMAND           "${GIT_EXECUTABLE}"
                      diff
                      HEAD
    OUTPUT_FILE       "${sprokit_binary_dir}/sprokit-${sprokit_version}${snap_suffix}${dirty_suffix}.patch"
    WORKING_DIRECTORY "${sprokit_source_dir}")
endif ()
