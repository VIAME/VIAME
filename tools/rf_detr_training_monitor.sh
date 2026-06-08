#!/bin/bash
# This file is part of VIAME, and is distributed under an OSI-approved        #
# BSD 3-Clause License. See top-level LICENSE.txt for details.                #
#
# Detached monitor for an RF-DETR (PyTorch-Lightning) slurm training job.
# Logs periodic status to a file and emails on: startup (test), an error/
# deadlock, normal completion, and every N epochs (with validation stats).
# Exits once the monitored job is no longer in the slurm queue.
#
# Configure via environment variables:
#   TM_JOBID          slurm job id (or set TM_JOBID_FILE to a file holding it)
#   TM_OUTPUT_DIR     rf-detr output dir containing metrics.csv      (required)
#   TM_LOG            training stdout/stderr log file                (required)
#   TM_EMAIL          recipient address                              (required)
#   TM_STATUS_LOG     status trail file        (default OUTPUT_DIR/monitor_status.log)
#   TM_EPOCH_INTERVAL email validation stats every N epochs          (default 10)
#   TM_POLL_SECONDS   poll period                                    (default 1200)
#   TM_DONE_PATTERN   log substring meaning "finished normally"  (default "TRAIN DONE (exit 0)")
#   TM_SENDMAIL       sendmail binary                                (default /usr/sbin/sendmail)
#
# Launch detached so it survives logout:
#   setsid nohup env TM_...=... bash rf_detr_training_monitor.sh >/dev/null 2>&1 &
set -u

JOBID="${TM_JOBID:-$(cat "${TM_JOBID_FILE:-/dev/null}" 2>/dev/null)}"
OUTPUT_DIR="${TM_OUTPUT_DIR:?set TM_OUTPUT_DIR}"
LOG="${TM_LOG:?set TM_LOG}"
EMAIL="${TM_EMAIL:?set TM_EMAIL}"
STATUS_LOG="${TM_STATUS_LOG:-$OUTPUT_DIR/monitor_status.log}"
EPOCH_INTERVAL="${TM_EPOCH_INTERVAL:-10}"
POLL="${TM_POLL_SECONDS:-1200}"
DONE_PATTERN="${TM_DONE_PATTERN:-TRAIN DONE (exit 0)}"
SENDMAIL="${TM_SENDMAIL:-/usr/sbin/sendmail}"
M="$OUTPUT_DIR/metrics.csv"
HOST=$(hostname)

send_email() {
  "$SENDMAIL" -t <<MAILEOF
To: ${EMAIL}
From: rf-detr-monitor@${HOST}
Subject: $1

$2
MAILEOF
}

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

status_block() {
  echo "job:       $JOBID  (host ${HOST})"
  echo "elapsed:   $(squeue -j "$JOBID" -h -o %M 2>/dev/null)"
  echo "max step:  $(tail -1 "$M" 2>/dev/null | cut -d, -f1-2)"
  echo "disk free: $(df -BG --output=avail "$OUTPUT_DIR" 2>/dev/null | tail -1 | tr -d ' ')"
  echo "latest validation:"
  val_stats latest
}

echo "$(date '+%F %T') monitor started for job $JOBID" >> "$STATUS_LOG"
send_email "[RF-DETR monitor] started (job $JOBID) — test email" \
"This confirms email alerts are working.

$(status_block)"

emailed_error=0
last_reported_epoch=-1
while true; do
  ts=$(date '+%F %T')
  if ! squeue -j "$JOBID" 2>/dev/null | grep -q "$JOBID"; then
    if grep -q "$DONE_PATTERN" "$LOG" 2>/dev/null; then
      subj="[RF-DETR monitor] job $JOBID FINISHED normally"; reason="COMPLETED_OK"
    else
      subj="[RF-DETR monitor] job $JOBID ENDED unexpectedly"; reason="ENDED_UNEXPECTEDLY"
    fi
    echo "$ts JOB NO LONGER RUNNING: $reason" >> "$STATUS_LOG"
    send_email "$subj" \
"reason: $reason

$(status_block)

best model: $OUTPUT_DIR/checkpoint_best_ema.pth
last log lines:
$(tail -20 "$LOG" 2>/dev/null | grep -ivE 'AccumulateGrad|run_backward|FutureWarning')"
    break
  fi

  step=$(tail -1 "$M" 2>/dev/null | cut -d, -f1-2)
  map=$(grep -iE 'Best EMA mAP' "$LOG" 2>/dev/null | tail -1 | sed 's/.*rf-detr - //')
  err=$(grep -iE 'watchdog|collective.*timeout|DistBackendError|Aborted \(core' "$LOG" 2>/dev/null | tail -1)
  free=$(df -BG --output=avail "$OUTPUT_DIR" 2>/dev/null | tail -1 | tr -d ' ')
  echo "$ts step=$step | ${map:-no-eval-yet} | disk_free=$free${err:+ | ERROR: $err}" >> "$STATUS_LOG"

  if [ -n "$err" ] && [ "$emailed_error" = 0 ]; then
    send_email "[RF-DETR monitor] job $JOBID ERROR detected" \
"A possible deadlock/error was detected:
$err

$(status_block)"
    emailed_error=1
  fi

  # Every N epochs: email validation statistics (1-indexed: epochs 10,20,...).
  ve=$(latest_val_epoch)
  if [ "$ve" -ge 0 ] && [ "$ve" != "$last_reported_epoch" ] \
       && [ $(((ve + 1) % EPOCH_INTERVAL)) -eq 0 ]; then
    send_email "[RF-DETR monitor] job $JOBID — validation at epoch $ve" \
"$(val_stats "$ve")

$(status_block)"
    last_reported_epoch=$ve
  fi

  sleep "$POLL"
done
