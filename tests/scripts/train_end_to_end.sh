#!/bin/bash

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# Train every tracker on a handful of clips for one epoch, twice, and check
# what came out.
#
#   tests/scripts/train_end_to_end.sh -i /path/to/annotated/clips
#
# This is not a unit test. It needs a GPU, a built install and real annotated
# data, so it is not wired into ctest; it is the thing to run before committing
# a training run that will occupy a machine for a day.
#
# The second pass is the point of it. Every stage here is guarded by a
# "already done, skip it" check that only executes on a resume, so a fresh run
# exercises none of them -- and those guards are where the failures have
# actually been. One tested for the wrong artifact and re-ran a stage that had
# already succeeded; another tested for an artifact that a *later* stage
# writes, and so skipped extraction that had never run at all, leaving the
# models after it training on features that were not there. Both are invisible
# on pass one and obvious on pass two.
#
# What it checks, per tracker:
#
#   pass 1   the run exits clean and leaves a model file behind
#   pass 2   with resume on, every stage reports that it is skipping, no
#            stage repeats its work, and the run still exits clean
#
# Options:
#   -i DIR     annotated clips, VIAME CSV beside each (required)
#   -o DIR     where to work (default: a temporary directory, removed after)
#   -t LIST    trackers to run (default: all six)
#   -n COUNT   clips to use (default: 3)
#   -k         keep the working directory
#   -g LIST    CUDA_VISIBLE_DEVICES for the runs (default: 0)

set -u

INPUT_DIRECTORY=""
WORK_DIRECTORY=""
TRACKERS="bytetrack ocsort deepsort botsort srnn siammask"
CLIP_COUNT=3
KEEP=false
GPUS="0"

while getopts "i:o:t:n:g:kh" OPTION
do
  case "${OPTION}" in
    i) INPUT_DIRECTORY="${OPTARG}" ;;
    o) WORK_DIRECTORY="${OPTARG}" ;;
    t) TRACKERS="${OPTARG}" ;;
    n) CLIP_COUNT="${OPTARG}" ;;
    g) GPUS="${OPTARG}" ;;
    k) KEEP=true ;;
    h) sed -n '7,38p' "$0" | cut -c3- ; exit 0 ;;
    *) exit 1 ;;
  esac
done

if [ -z "${INPUT_DIRECTORY}" ]
then
  echo "ERROR: -i is required. See -h."
  exit 1
fi

if [ ! -d "${INPUT_DIRECTORY}" ]
then
  echo "ERROR: no such input directory: ${INPUT_DIRECTORY}"
  exit 1
fi

if ! command -v viame > /dev/null 2>&1
then
  echo "ERROR: viame is not on the path. Source setup_viame.sh first."
  exit 1
fi

# The install root, for the config files that live beside the build rather
# than in the source tree. Taken from the binary actually being used, so this
# stays right when more than one install is present.
VIAME_INSTALL="${VIAME_INSTALL:-$( dirname "$( dirname "$( command -v viame )" )" )}"

# One epoch everywhere, unless the caller asked for more. Each of these is
# read by the trainer it belongs to; the point is to reach every stage, not
# to fit anything -- but a caller proving epoch-dependent behaviour (early
# stopping, LR steps) needs its own values to survive, and an unconditional
# export here silently clobbered them.
export VIAME_SRNN_SIAMESE_EPOCHS="${VIAME_SRNN_SIAMESE_EPOCHS:-1}"
export VIAME_SRNN_LSTM_EPOCHS="${VIAME_SRNN_LSTM_EPOCHS:-1}"

if [ -z "${WORK_DIRECTORY}" ]
then
  WORK_DIRECTORY=$( mktemp -d -t viame-e2e-XXXXXX )
  CREATED_WORK_DIRECTORY=true
else
  mkdir -p "${WORK_DIRECTORY}"
  CREATED_WORK_DIRECTORY=false
fi

cleanup()
{
  if [ "${KEEP}" = true ] || [ "${CREATED_WORK_DIRECTORY}" = false ]
  then
    echo
    echo "Working directory kept at ${WORK_DIRECTORY}"
  else
    rm -rf "${WORK_DIRECTORY}"
  fi
}

trap cleanup EXIT

FIXTURE="${WORK_DIRECTORY}/clips"
mkdir -p "${FIXTURE}"

# ---------------------------------------------------------------------------
# The fixture: the smallest clips that carry annotations
# ---------------------------------------------------------------------------

# Smallest by frame count, because everything downstream scales with it and a
# fixture that takes an hour will not get run. Symlinked rather than copied:
# the image folders are the bulk of the data and nothing here writes to them.
echo "Selecting ${CLIP_COUNT} clip(s) from ${INPUT_DIRECTORY}"

SELECTED=$( python3 - "${INPUT_DIRECTORY}" "${CLIP_COUNT}" <<'PY'
import os
import sys

directory, wanted = sys.argv[1], int(sys.argv[2])
candidates = []

for name in sorted(os.listdir(directory)):
    folder = os.path.join(directory, name)

    if not os.path.isdir(folder):
        continue

    # A clip is usable here only with groundtruth beside it, either inside the
    # folder or named for it alongside.
    truth = None

    for candidate in (os.path.join(directory, name + '.csv'),):
        if os.path.isfile(candidate):
            truth = candidate

    if truth is None:
        inside = [f for f in os.listdir(folder) if f.endswith('.csv')]

        if inside:
            truth = os.path.join(folder, inside[0])

    if truth is None:
        continue

    frames = len([f for f in os.listdir(folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])

    if not frames:
        continue

    # Track count, because a clip holding one track is useless to half of
    # these trainers: a Siamese pair needs a detection outside its own track
    # to contrast against, and re-identification needs two identities to tell
    # apart. A fixture built from single track clips fails in the trainer
    # rather than in anything it was meant to test.
    identifiers = set()

    try:
        with open(truth) as handle:
            for line in handle:
                if line.startswith('#') or not line.strip():
                    continue

                fields = line.split(',')

                if fields:
                    identifiers.add(fields[0].strip())
    except OSError:
        pass

    candidates.append((len(identifiers) < 2, frames, name, truth,
                       len(identifiers)))

# Multi track clips first, then smallest. Single track clips are still used
# if there are not enough of the others, since some trainers do fine on them.
for _, frames, name, truth, tracks in sorted(candidates)[:wanted]:
    print('{}\t{}\t{} frames, {} tracks'.format(name, truth, frames, tracks))
PY
)

if [ -z "${SELECTED}" ]
then
  echo "ERROR: found no clip folders with groundtruth in ${INPUT_DIRECTORY}"
  exit 1
fi

while IFS=$'\t' read -r NAME TRUTH FRAMES
do
  ln -sfn "${INPUT_DIRECTORY}/${NAME}" "${FIXTURE}/${NAME}"
  ln -sfn "${TRUTH}" "${FIXTURE}/${NAME}.csv"
  echo "  ${NAME} (${FRAMES})"
done <<< "${SELECTED}"

# ---------------------------------------------------------------------------
# Settings, matching what a real run writes
# ---------------------------------------------------------------------------

write_settings()
{
  local TRACKER="$1"
  local SETTINGS="$2"
  local RESUME="$3"

  cat > "${SETTINGS}" <<EOF
image_reader:type=ocv
groundtruth_reader:type=auto
groundtruth_style=one_per_folder
groundtruth_extensions=.csv;.json;.kw18
track_reader:type=viame_csv
track_reader:viame_csv:batch_load=true
output_directory=category_models
detector_trainer:type=svm
detector_trainer:svm:ingest_pipeline=pipelines/index_fish.svm.pipe
EOF

  case "${TRACKER}" in
    srnn)
      echo "tracker_trainer:srnn:resume=${RESUME}" >> "${SETTINGS}"
      echo "tracker_trainer:srnn:lstm_concurrency=1" >> "${SETTINGS}"
      ;;
    siammask|siamrpn)
      # The architecture config, which the standard training config supplies
      # as a relative path and which the trainer has no working fallback for.
      local ARCHITECTURE="${VIAME_INSTALL}/configs/pipelines/models/siammask_default.yaml"

      if [ -f "${ARCHITECTURE}" ]
      then
        echo "tracker_trainer:${TRACKER}:config_file=${ARCHITECTURE}" \
          >> "${SETTINGS}"
      fi

      # Few enough samples that an epoch is a minute, and the polygons
      # rasterised so the mask head is actually supervised.
      echo "tracker_trainer:${TRACKER}:samples_per_sequence=20" >> "${SETTINGS}"
      echo "tracker_trainer:${TRACKER}:max_epochs=1" >> "${SETTINGS}"
      # Rasterise the groundtruth polygons as the CSV is read, so the mask
      # head is supervised rather than silently trained on nothing.
      echo "track_reader:viame_csv:poly_to_mask=true" >> "${SETTINGS}"
      ;;
    deepsort|botsort)
      echo "tracker_trainer:${TRACKER}:max_epochs=1" >> "${SETTINGS}"
      ;;
  esac
}

run_pass()
{
  local TRACKER="$1"
  local RUN_DIRECTORY="$2"
  local RESUME="$3"
  local LOG="$4"

  local SETTINGS="${RUN_DIRECTORY}/settings.conf"

  mkdir -p "${RUN_DIRECTORY}"
  write_settings "${TRACKER}" "${SETTINGS}" "${RESUME}"

  (
    cd "${RUN_DIRECTORY}" && \
    CUDA_VISIBLE_DEVICES="${GPUS}" viame train \
      -i "${FIXTURE}" \
      --tracker "${TRACKER}" \
      --settings-file "${SETTINGS}" \
      --no-query \
      --llm-assist off
  ) > "${LOG}" 2>&1

  return $?
}

# Did it train, and is there something to show for it.
#
# Both halves are needed. The success line alone passes when a trainer reports
# success and writes nothing, and a file alone passes on a stray log. The file
# has to be newer than the settings written just before the run, so a model
# left by the previous pass does not stand in for this one.
trained_ok()
{
  grep -q "Tracker training completed successfully" "$1"
}

model_present()
{
  local RUN_DIRECTORY="$1"

  # Anything a trainer here writes: torch weights, an svm, a params file, a
  # pickled model. Not settings.conf itself, which is the reference point.
  find "${RUN_DIRECTORY}" -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.svm' -o -name '*.json' \
       -o -name '*.pkl' -o -name '*.p' -o -name '*.zip' \) \
    -newer "${RUN_DIRECTORY}/settings.conf" 2>/dev/null | grep -q .
}

# ---------------------------------------------------------------------------

echo
echo "=========================================================="
echo "End to end over ${CLIP_COUNT} clip(s), one epoch, on GPU ${GPUS}"
echo "  trackers  ${TRACKERS}"
echo "  work      ${WORK_DIRECTORY}"
echo "=========================================================="

FAILURES=0
PASSED=""
FAILED=""

for TRACKER in ${TRACKERS}
do
  RUN_DIRECTORY="${WORK_DIRECTORY}/${TRACKER}"
  FIRST_LOG="${WORK_DIRECTORY}/${TRACKER}-pass1.log"
  SECOND_LOG="${WORK_DIRECTORY}/${TRACKER}-pass2.log"

  echo
  echo "[$( date +%H:%M:%S )] ${TRACKER}: pass 1, from nothing"

  run_pass "${TRACKER}" "${RUN_DIRECTORY}" false "${FIRST_LOG}"
  FIRST_STATUS=$?

  TROUBLE=""

  if [ ${FIRST_STATUS} -ne 0 ]
  then
    TROUBLE="pass 1 exited ${FIRST_STATUS}"
  elif grep -qiE '^(Traceback|.*\bError\b.*)' "${FIRST_LOG}" \
       && grep -qi 'Traceback' "${FIRST_LOG}"
  then
    TROUBLE="pass 1 raised"
  elif ! trained_ok "${FIRST_LOG}"
  then
    TROUBLE="pass 1 never reported success"
  elif ! model_present "${RUN_DIRECTORY}"
  then
    TROUBLE="pass 1 reported success but wrote no model"
  fi

  if [ -n "${TROUBLE}" ]
  then
    echo "  FAIL: ${TROUBLE}"
    grep -iE 'error|Traceback|Exception' "${FIRST_LOG}" | tail -5 | sed 's/^/    /'
    echo "    log: ${FIRST_LOG}"
    KEEP=true
    FAILURES=$(( FAILURES + 1 ))
    FAILED="${FAILED} ${TRACKER}"
    continue
  fi

  echo "  pass 1 ok"

  # Pass two, on top of pass one's output. Only srnn has staged resume guards
  # today; the others are re-run anyway, because a second run over a populated
  # directory is its own check -- that is where the tree removal that assumed
  # a directory it had just deleted still existed came from.
  echo "[$( date +%H:%M:%S )] ${TRACKER}: pass 2, resuming"

  run_pass "${TRACKER}" "${RUN_DIRECTORY}" true "${SECOND_LOG}"
  SECOND_STATUS=$?

  if [ ${SECOND_STATUS} -ne 0 ]
  then
    echo "  FAIL: pass 2 exited ${SECOND_STATUS}"
    grep -iE 'error|Traceback|Exception' "${SECOND_LOG}" | tail -5 | sed 's/^/    /'
    echo "    log: ${SECOND_LOG}"
    KEEP=true
    FAILURES=$(( FAILURES + 1 ))
    FAILED="${FAILED} ${TRACKER}"
    continue
  fi

  if ! trained_ok "${SECOND_LOG}"
  then
    echo "  FAIL: pass 2 never reported success"
    grep -iE 'error|Traceback|Exception' "${SECOND_LOG}" | tail -5 | sed 's/^/    /'
    echo "    log: ${SECOND_LOG}"
    KEEP=true
    FAILURES=$(( FAILURES + 1 ))
    FAILED="${FAILED} ${TRACKER}"
    continue
  fi

  if [ "${TRACKER}" = "srnn" ]
  then
    # Each of the six stages announces itself and, when resuming, says it is
    # skipping. A stage that announces itself without skipping did its work
    # twice, which means its guard did not fire.
    STAGES_RUN=$( grep -cE '^(Creating Siamese training data|Training Siamese model|Extracting appearance features|Creating LSTM training data|Training .* individual LSTM models|Training combined LSTM model)' "${SECOND_LOG}" )
    STAGES_SKIPPED=$( grep -ciE 'skipping|already trained|already generated|already extracted' "${SECOND_LOG}" )

    echo "    ${STAGES_RUN} stage(s) reached, ${STAGES_SKIPPED} skip(s) reported"

    if [ "${STAGES_SKIPPED}" -lt "${STAGES_RUN}" ]
    then
      echo "  FAIL: a stage repeated work on resume"
      echo "        every stage reached should report skipping"
      grep -nE '^(Creating|Training|Extracting)' "${SECOND_LOG}" \
        | tail -12 | sed 's/^/    /'
      echo "    log: ${SECOND_LOG}"
      KEEP=true
      FAILURES=$(( FAILURES + 1 ))
      FAILED="${FAILED} ${TRACKER}"
      continue
    fi
  fi

  echo "  pass 2 ok"
  PASSED="${PASSED} ${TRACKER}"
done

echo
echo "=========================================================="

if [ ${FAILURES} -eq 0 ]
then
  echo "All clear:${PASSED}"
else
  echo "Passed:${PASSED:- none}"
  echo "Failed:${FAILED}"
fi

echo "=========================================================="

exit ${FAILURES}
