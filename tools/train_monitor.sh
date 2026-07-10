#!/bin/bash
# This file is part of VIAME, and is distributed under an OSI-approved        #
# BSD 3-Clause License. See top-level LICENSE.txt for details.                #
#
# Detached monitor for a VIAME slurm training job. Logs periodic status to a
# file and emails on: startup (test), a detected error/deadlock, normal
# completion, unexpected end, and periodic progress (per-epoch validation for
# RF-DETR, stage/heartbeat updates for netharn). Exits once the monitored job
# is no longer in the slurm queue.
#
# Supports two training types (TM_TYPE, default auto):
#   * rf_detr  -- PyTorch-Lightning RF-DETR; reads metrics.csv for validation
#                 stats and emails every N epochs.
#   * netharn  -- netharn detector/classifier/refiner (e.g. the sea lion
#                 reclassifier); parses the training log for the current stage
#                 (background mining -> training -> done) and latest epoch/loss,
#                 and emails on stage changes plus a periodic heartbeat.
#   * auto     -- rf_detr if TM_OUTPUT_DIR/metrics.csv exists, else netharn.
#
# Configure via environment variables:
#   TM_JOBID          slurm job id (or set TM_JOBID_FILE to a file holding it)
#   TM_OUTPUT_DIR     output/run dir (status log + disk-free live here)  (required)
#   TM_LOG            training stdout/stderr log file                    (required)
#   TM_EMAIL          recipient address                                  (required)
#   TM_TYPE           auto | rf_detr | netharn                    (default auto)
#   TM_JOB_NAME       label used in email subjects                (default TM_TYPE)
#   TM_STATUS_LOG     status trail file        (default OUTPUT_DIR/monitor_status.log)
#   TM_EPOCH_INTERVAL email validation/epoch stats every N epochs        (default 5)
#   TM_HEARTBEAT_POLLS netharn: send a heartbeat every N polls           (default 6)
#   TM_CHIP_DIR       netharn: chip dir to report progress from          (optional)
#   TM_POLL_SECONDS   poll period                                    (default 1200)
#   TM_DONE_PATTERN   log substring meaning "finished normally"
#                                             (default "TRAIN DONE (exit 0)")
#   TM_METRICS_CSV    rf_detr: metrics.csv path      (default OUTPUT_DIR/metrics.csv)
#   TM_SENDMAIL       sendmail binary                       (default /usr/sbin/sendmail)
#
# Launch detached so it survives logout:
#   setsid nohup env TM_...=... bash train_monitor.sh >/dev/null 2>&1 &
set -u

JOBID="${TM_JOBID:-$(cat "${TM_JOBID_FILE:-/dev/null}" 2>/dev/null)}"
OUTPUT_DIR="${TM_OUTPUT_DIR:?set TM_OUTPUT_DIR}"
LOG="${TM_LOG:?set TM_LOG}"
EMAIL="${TM_EMAIL:?set TM_EMAIL}"
STATUS_LOG="${TM_STATUS_LOG:-$OUTPUT_DIR/monitor_status.log}"
EPOCH_INTERVAL="${TM_EPOCH_INTERVAL:-5}"
HEARTBEAT_POLLS="${TM_HEARTBEAT_POLLS:-6}"
CHIP_DIR="${TM_CHIP_DIR:-}"
POLL="${TM_POLL_SECONDS:-1200}"
DONE_PATTERN="${TM_DONE_PATTERN:-TRAIN DONE (exit 0)}"
METRICS_CSV="${TM_METRICS_CSV:-$OUTPUT_DIR/metrics.csv}"
SENDMAIL="${TM_SENDMAIL:-/usr/sbin/sendmail}"
HOST=$(hostname)

# Resolve training type.
TYPE="${TM_TYPE:-auto}"
if [ "$TYPE" = "auto" ]; then
  if [ -f "$METRICS_CSV" ]; then TYPE=rf_detr; else TYPE=netharn; fi
fi
JOB_NAME="${TM_JOB_NAME:-$TYPE}"
M="$METRICS_CSV"

send_email() {
  "$SENDMAIL" -t <<MAILEOF
To: ${EMAIL}
From: train-monitor@${HOST}
Subject: $1

$2
MAILEOF
}

# ------------------------------------------------------------------ rf_detr ---
# Print validation stats for a given epoch ("latest" or a number) from metrics.csv.
val_stats() {
  python3 - "$M" "$1" <<'PY'
import csv, sys
path, want = sys.argv[1], sys.argv[2]
try:
    rows = list(csv.DictReader(open(path)))
except Exception:
    print("  (metrics.csv not readable yet)"); sys.exit()
val = [r for r in rows if r.get('val/mAP_50_95', '')]
if not val:
    print("  (no completed validation yet)"); sys.exit()
if want == 'latest':
    r = val[-1]
else:
    cand = [x for x in val if x.get('epoch') == str(want)]
    r = cand[-1] if cand else val[-1]
def g(k):
    v = r.get(k, '')
    try: return f"{float(v):.4f}"
    except (TypeError, ValueError): return "n/a"
print(f"  epoch:          {r.get('epoch')}")
print(f"  mAP@50:95:      {g('val/mAP_50_95')}   (EMA {g('val/ema_mAP_50_95')})")
print(f"  mAP@50:         {g('val/mAP_50')}    mAP@75: {g('val/mAP_75')}")
print(f"  F1/Prec/Recall: {g('val/F1')} / {g('val/precision')} / {g('val/recall')}")
print("  per-class AP@50:95:")
for k in sorted(r):
    if k.startswith('val/AP/'):
        print(f"    {k[7:]:18s} {g(k)}")
PY
}

# Highest epoch that has completed validation (a row with val/mAP_50_95 set).
latest_val_epoch() {
  python3 - "$M" <<'PY'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
except Exception:
    print(-1); sys.exit()
ep = [int(r['epoch']) for r in rows if r.get('val/mAP_50_95', '') and r.get('epoch','').isdigit()]
print(max(ep) if ep else -1)
PY
}

# ------------------------------------------------------------------ netharn ---
# Best-effort current training stage from the log.
nh_stage() {
  if grep -q "$DONE_PATTERN" "$LOG" 2>/dev/null; then echo "done"; return; fi
  # netharn per-epoch progress lines contain "vloss"; chip mining prints
  # "Time to Detect Objects".
  if tr '\r' '\n' < "$LOG" 2>/dev/null | grep -qiE 'vloss|Maximum .*epoch|Fitting|best=' ; then
    echo "training"
  elif grep -qiE 'Time to Detect Objects|Extracting|writing chip|image_chips' "$LOG" 2>/dev/null; then
    echo "mining background / extracting chips"
  elif grep -qiE 'Read [0-9]+ detections|Opening groundtruth|Processing ' "$LOG" 2>/dev/null; then
    echo "loading groundtruth"
  else
    echo "initializing"
  fi
}

# Latest netharn epoch integer seen in the log (-1 if none yet).
nh_epoch() {
  tr '\r' '\n' < "$LOG" 2>/dev/null | python3 - <<'PY'
import re, sys
ep = -1
for line in sys.stdin:
    for m in re.finditer(r'epoch[^0-9]{0,4}(\d+)', line, re.I):
        try: ep = max(ep, int(m.group(1)))
        except ValueError: pass
print(ep)
PY
}

# Latest netharn epoch/loss line for the email body.
nh_latest_line() {
  tr '\r' '\n' < "$LOG" 2>/dev/null | grep -iE 'vloss|best=|Maximum .*epoch' | tail -1
}

# ------------------------------------------------------------------- shared ---
status_block() {
  echo "job:       $JOBID  (host ${HOST}, type ${TYPE})"
  echo "elapsed:   $(squeue -j "$JOBID" -h -o %M 2>/dev/null)"
  echo "stage:     $STAGE"
  if [ "$TYPE" = "rf_detr" ]; then
    echo "max step:  $(tail -1 "$M" 2>/dev/null | cut -d, -f1-2)"
  fi
  if [ -n "$CHIP_DIR" ]; then
    echo "chips:     $(find "$CHIP_DIR" -type f 2>/dev/null | wc -l)"
  fi
  echo "disk free: $(df -BG --output=avail "$OUTPUT_DIR" 2>/dev/null | tail -1 | tr -d ' ')"
  if [ "$TYPE" = "rf_detr" ]; then
    echo "latest validation:"
    val_stats latest
  else
    local ln; ln=$(nh_latest_line)
    echo "latest epoch line: ${ln:-<none yet>}"
  fi
}

STAGE=$(nh_stage 2>/dev/null || echo "n/a")
echo "$(date '+%F %T') monitor started for job $JOBID (type $TYPE)" >> "$STATUS_LOG"
send_email "[train-monitor:${JOB_NAME}] started (job $JOBID) — test email" \
"This confirms email alerts are working.

$(status_block)"

emailed_error=0
last_reported_epoch=-1
last_stage="$STAGE"
polls_since_beat=0
while true; do
  ts=$(date '+%F %T')
  if ! squeue -j "$JOBID" 2>/dev/null | grep -q "$JOBID"; then
    STAGE=$(nh_stage)
    if grep -q "$DONE_PATTERN" "$LOG" 2>/dev/null; then
      subj="[train-monitor:${JOB_NAME}] job $JOBID FINISHED normally"; reason="COMPLETED_OK"
    else
      subj="[train-monitor:${JOB_NAME}] job $JOBID ENDED unexpectedly"; reason="ENDED_UNEXPECTEDLY"
    fi
    echo "$ts JOB NO LONGER RUNNING: $reason" >> "$STATUS_LOG"
    send_email "$subj" \
"reason: $reason

$(status_block)

last log lines:
$(tail -25 "$LOG" 2>/dev/null | grep -ivE 'AccumulateGrad|run_backward|FutureWarning|schedule_dep')"
    break
  fi

  STAGE=$(nh_stage)
  free=$(df -BG --output=avail "$OUTPUT_DIR" 2>/dev/null | tail -1 | tr -d ' ')
  err=$(grep -iE 'watchdog|collective.*timeout|DistBackendError|Aborted \(core|CUDA out of memory|Traceback \(most recent' "$LOG" 2>/dev/null | tail -1)

  if [ "$TYPE" = "rf_detr" ]; then
    step=$(tail -1 "$M" 2>/dev/null | cut -d, -f1-2)
    map=$(grep -iE 'Best EMA mAP' "$LOG" 2>/dev/null | tail -1 | sed 's/.*rf-detr - //')
    echo "$ts step=$step | ${map:-no-eval-yet} | disk_free=$free${err:+ | ERROR: $err}" >> "$STATUS_LOG"
  else
    chips=""; [ -n "$CHIP_DIR" ] && chips="chips=$(find "$CHIP_DIR" -type f 2>/dev/null | wc -l) | "
    echo "$ts stage=$STAGE | ${chips}$(nh_latest_line) | disk_free=$free${err:+ | ERROR: $err}" >> "$STATUS_LOG"
  fi

  # Error alert (once).
  if [ -n "$err" ] && [ "$emailed_error" = 0 ]; then
    send_email "[train-monitor:${JOB_NAME}] job $JOBID ERROR detected" \
"A possible error/deadlock was detected:
$err

$(status_block)"
    emailed_error=1
  fi

  # Stage-change alert (netharn: e.g. mining -> training).
  if [ "$STAGE" != "$last_stage" ]; then
    send_email "[train-monitor:${JOB_NAME}] job $JOBID — stage: $STAGE" \
"Training stage changed: $last_stage -> $STAGE

$(status_block)"
    last_stage="$STAGE"
    polls_since_beat=0
  fi

  if [ "$TYPE" = "rf_detr" ]; then
    # Every N epochs: email validation statistics (1-indexed: epochs 10,20,...).
    ve=$(latest_val_epoch)
    if [ "$ve" -ge 0 ] && [ "$ve" != "$last_reported_epoch" ] \
         && [ $(((ve + 1) % EPOCH_INTERVAL)) -eq 0 ]; then
      send_email "[train-monitor:${JOB_NAME}] job $JOBID — validation at epoch $ve" \
"$(val_stats "$ve")

$(status_block)"
      last_reported_epoch=$ve
    fi
  else
    # netharn: per-epoch email if a new interval epoch is reached, else a
    # periodic heartbeat so status still arrives during long stages.
    ve=$(nh_epoch)
    polls_since_beat=$((polls_since_beat + 1))
    if [ "$ve" -ge 0 ] && [ "$ve" != "$last_reported_epoch" ] \
         && [ $((ve % EPOCH_INTERVAL)) -eq 0 ]; then
      send_email "[train-monitor:${JOB_NAME}] job $JOBID — epoch $ve" \
"$(status_block)"
      last_reported_epoch=$ve
      polls_since_beat=0
    elif [ "$polls_since_beat" -ge "$HEARTBEAT_POLLS" ]; then
      send_email "[train-monitor:${JOB_NAME}] job $JOBID — heartbeat ($STAGE)" \
"$(status_block)"
      polls_since_beat=0
    fi
  fi

  sleep "$POLL"
done
