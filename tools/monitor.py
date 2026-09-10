#!/usr/bin/env python3
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""monitor.py - Watch a VIAME training run and report on its progress.

A detached monitor periodically inspects a training log (and, for RF-DETR
runs, its metrics.csv) and records a status trail in the run's output
folder. It can also send email on startup (a test message), on a detected
error or deadlock, on training stage changes, on periodic epoch or
heartbeat milestones, and when the run ends (normally or otherwise).

Two training types are recognised (--type, default auto):

  rf_detr   PyTorch-Lightning RF-DETR; validation statistics are read from
            metrics.csv and reported every N epochs.
  netharn   netharn detector/classifier/refiner; the training log is parsed
            for the current stage (loading groundtruth -> mining chips ->
            training -> done) and latest epoch/loss, with reports on stage
            changes plus a periodic heartbeat.
  auto      rf_detr when metrics.csv exists in the output folder, netharn
            otherwise.

The run is considered alive while, in order of preference, its slurm job is
queued (--job-id), its process exists (--pid), or its log file has changed
within --stale-minutes. Email goes through an SMTP server (--smtp-server,
or the VIAME_SMTP_SERVER environment variable) or, on Linux and macOS, a
local sendmail; without either the reports are only written to the status
trail and monitor output file.

Examples:
  viame monitor start -o runs/seals -l runs/seals/train.log --job-id 12345 \\
      --email me@example.com
  viame monitor start -o runs/seals -l runs/seals/train.log --pid 4242 \\
      --email me@example.com --smtp-server smtp.example.com:587 --smtp-user me
  viame monitor status runs/seals
  viame monitor stop runs/seals
"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import shutil
import signal
import smtplib
import socket
import subprocess
import sys
import time
from email.message import EmailMessage

SETTINGS_FILE = 'monitor.json'
PID_FILE = 'monitor.pid'
OUTPUT_FILE = 'monitor.out'
DEFAULT_STATUS_LOG = 'monitor_status.log'
DEFAULT_DONE_PATTERN = 'TRAIN DONE (exit 0)'

ERROR_RE = re.compile(
    r'watchdog|collective.*timeout|DistBackendError|Aborted \(core|'
    r'CUDA out of memory|Traceback \(most recent', re.I)
NOISE_RE = re.compile(r'AccumulateGrad|run_backward|FutureWarning|schedule_dep', re.I)
NH_TRAINING_RE = re.compile(r'vloss|Maximum .*epoch|Fitting|best=', re.I)
NH_MINING_RE = re.compile(r'Time to Detect Objects|Extracting|writing chip|image_chips', re.I)
NH_LOADING_RE = re.compile(r'Read [0-9]+ detections|Opening groundtruth|Processing ', re.I)
NH_EPOCH_RE = re.compile(r'epoch[^0-9]{0,4}(\d+)', re.I)
RF_BEST_MAP_RE = re.compile(r'Best EMA mAP', re.I)


def timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def read_text(path):
    """Whole file as text with carriage-return progress lines split out."""
    try:
        with open(path, 'r', errors='replace') as fin:
            return fin.read().replace('\r', '\n')
    except OSError:
        return ''


def tail_lines(path, count):
    lines = read_text(path).splitlines()
    return lines[-count:]


def disk_free(path):
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        return 'n/a'
    return '%dG' % (free // (1024 ** 3))


def count_files(folder):
    total = 0
    for _root, _dirs, files in os.walk(folder):
        total += len(files)
    return total


# ----------------------------------------------------------------- rf_detr ---

def read_metrics(path):
    try:
        with open(path, newline='') as fin:
            return list(csv.DictReader(fin))
    except OSError:
        return None


def val_rows(rows):
    return [r for r in rows if r.get('val/mAP_50_95', '')]


def val_stats(metrics_csv, want='latest'):
    """Validation statistics for an epoch ('latest' or a number), as text."""
    rows = read_metrics(metrics_csv)
    if rows is None:
        return '  (metrics.csv not readable yet)'
    val = val_rows(rows)
    if not val:
        return '  (no completed validation yet)'
    if want == 'latest':
        row = val[-1]
    else:
        cand = [r for r in val if r.get('epoch') == str(want)]
        row = cand[-1] if cand else val[-1]

    def g(key):
        try:
            return '%.4f' % float(row.get(key, ''))
        except (TypeError, ValueError):
            return 'n/a'

    out = [
        '  epoch:          %s' % row.get('epoch'),
        '  mAP@50:95:      %s   (EMA %s)' % (g('val/mAP_50_95'), g('val/ema_mAP_50_95')),
        '  mAP@50:         %s    mAP@75: %s' % (g('val/mAP_50'), g('val/mAP_75')),
        '  F1/Prec/Recall: %s / %s / %s' % (g('val/F1'), g('val/precision'), g('val/recall')),
        '  per-class AP@50:95:',
    ]
    for key in sorted(row):
        if key.startswith('val/AP/'):
            out.append('    %-18s %s' % (key[7:], g(key)))
    return '\n'.join(out)


def latest_val_epoch(metrics_csv):
    """Highest epoch with completed validation, or -1."""
    rows = read_metrics(metrics_csv)
    if not rows:
        return -1
    epochs = [int(r['epoch']) for r in val_rows(rows)
              if r.get('epoch', '').isdigit()]
    return max(epochs) if epochs else -1


def latest_step(metrics_csv):
    lines = tail_lines(metrics_csv, 1)
    if not lines:
        return 'n/a'
    return ','.join(lines[-1].split(',')[:2])


def latest_best_map(log_text):
    hits = [ln for ln in log_text.splitlines() if RF_BEST_MAP_RE.search(ln)]
    if not hits:
        return 'no-eval-yet'
    return re.sub(r'.*rf-detr - ', '', hits[-1])


# ----------------------------------------------------------------- netharn ---

def nh_stage(log_text, done_pattern):
    if done_pattern in log_text:
        return 'done'
    if NH_TRAINING_RE.search(log_text):
        return 'training'
    if NH_MINING_RE.search(log_text):
        return 'mining background / extracting chips'
    if NH_LOADING_RE.search(log_text):
        return 'loading groundtruth'
    return 'initializing'


def nh_epoch(log_text):
    epoch = -1
    for line in log_text.splitlines():
        if re.search(r'max(?:imum|_)?\s*epochs?|max_epochs|epoch.*(?:limit|maximum)', line, re.I):
            continue
        match = NH_EPOCH_RE.search(line)
        if match:
            epoch = int(match.group(1))
    return epoch


def nh_latest_line(log_text):
    hits = [ln for ln in log_text.splitlines() if NH_TRAINING_RE.search(ln)]
    return hits[-1] if hits else ''


def last_error(log_text):
    hits = [ln for ln in log_text.splitlines() if ERROR_RE.search(ln)]
    return hits[-1] if hits else ''


# ---------------------------------------------------------------- liveness ---

def slurm_elapsed(job_id):
    try:
        out = subprocess.run(['squeue', '-j', str(job_id), '-h', '-o', '%M'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return out.stdout.strip()


def slurm_running(job_id):
    """True while the job is in the queue; None when squeue is unavailable."""
    try:
        out = subprocess.run(['squeue', '-j', str(job_id), '-h', '-o', '%i'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return str(job_id) in out.stdout.split()


def pid_running(pid):
    if sys.platform == 'win32':
        try:
            out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % pid, '/NH'],
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return str(pid) in out.stdout.split()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def log_recent(log, stale_minutes):
    try:
        age = time.time() - os.path.getmtime(log)
    except OSError:
        return False
    return age < stale_minutes * 60


# ------------------------------------------------------------------- email ---

class Notifier:
    """Send reports by SMTP, sendmail, or only to the status trail."""

    def __init__(self, args, status_log):
        self.args = args
        self.status_log = status_log
        self.host = socket.gethostname()
        self.sender = args.smtp_from or ('train-monitor@%s' % self.host)
        self.sendmail = None
        if not args.smtp_server and args.email:
            candidates = [args.sendmail] if args.sendmail else []
            candidates += ['/usr/sbin/sendmail', '/usr/lib/sendmail', 'sendmail']
            for cand in candidates:
                found = cand if os.path.isfile(cand) else shutil.which(cand)
                if found:
                    self.sendmail = found
                    break

    def describe(self):
        if not self.args.email:
            return 'no email configured; reports go to the status trail only'
        if self.args.smtp_server:
            return 'email via SMTP %s to %s' % (self.args.smtp_server, self.args.email)
        if self.sendmail:
            return 'email via %s to %s' % (self.sendmail, self.args.email)
        return ('no SMTP server and no sendmail found; reports for %s go to '
                'the status trail only' % self.args.email)

    def send(self, subject, body):
        append_status(self.status_log, 'REPORT: %s' % subject)
        print('[%s] %s\n%s\n' % (timestamp(), subject, body), flush=True)
        if not self.args.email:
            return
        try:
            if self.args.smtp_server:
                self._send_smtp(subject, body)
            elif self.sendmail:
                self._send_sendmail(subject, body)
        except Exception as err:  # never let a mail failure stop monitoring
            append_status(self.status_log, 'EMAIL FAILED: %s' % err)
            print('email failed: %s' % err, file=sys.stderr, flush=True)

    def _message(self, subject, body):
        msg = EmailMessage()
        msg['To'] = self.args.email
        msg['From'] = self.sender
        msg['Subject'] = subject
        msg.set_content(body)
        return msg

    def _send_smtp(self, subject, body):
        host, _, port = self.args.smtp_server.partition(':')
        port = int(port) if port else 587
        password = os.environ.get(self.args.smtp_password_env, '') \
            if self.args.smtp_password_env else ''
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=60)
        else:
            server = smtplib.SMTP(host, port, timeout=60)
        with server:
            if port != 465:
                try:
                    server.starttls()
                except smtplib.SMTPNotSupportedError:
                    pass
            if self.args.smtp_user:
                server.login(self.args.smtp_user, password)
            server.send_message(self._message(subject, body))

    def _send_sendmail(self, subject, body):
        msg = self._message(subject, body)
        subprocess.run([self.sendmail, '-t', '-oi'], input=msg.as_bytes(),
                       check=True, timeout=120)


def append_status(status_log, line):
    try:
        with open(status_log, 'a') as fout:
            fout.write('%s %s\n' % (timestamp(), line))
    except OSError as err:
        print('cannot write %s: %s' % (status_log, err), file=sys.stderr)


# ----------------------------------------------------------------- monitor ---

class Monitor:
    def __init__(self, args):
        self.args = args
        self.output_dir = os.path.abspath(args.output_dir)
        self.log = os.path.abspath(args.log)
        self.metrics_csv = args.metrics_csv or os.path.join(self.output_dir, 'metrics.csv')
        self.status_log = args.status_log or os.path.join(self.output_dir, DEFAULT_STATUS_LOG)
        self.started = getattr(args, 'started', None) or time.time()
        self.host = socket.gethostname()
        self.notifier = Notifier(args, self.status_log)
        self.job_label = self.job_description()
        self._log_offset = 0
        self._log_identity = None
        self._log_tail = ''

    def read_log(self):
        try:
            stat = os.stat(self.log)
            identity = (stat.st_dev, stat.st_ino)
            if identity != self._log_identity or stat.st_size < self._log_offset:
                self._log_offset = 0
                self._log_tail = ''
            with open(self.log, 'rb') as stream:
                # Bound both the initial read and bursts of new output.
                stream.seek(max(self._log_offset, stat.st_size - 2 * 1024 * 1024))
                chunk = stream.read()
                self._log_offset = stream.tell()
            self._log_identity = identity
            self._log_tail = (self._log_tail + chunk.decode(errors='replace').replace('\r', '\n'))[-2 * 1024 * 1024:]
        except OSError:
            pass
        return self._log_tail

    # -- run identity ----------------------------------------------------
    def job_description(self):
        if self.args.job_id:
            return 'job %s' % self.args.job_id
        if self.args.pid:
            return 'pid %d' % self.args.pid
        return os.path.basename(self.output_dir.rstrip(os.sep)) or 'run'

    def resolve_type(self):
        if self.args.type != 'auto':
            return self.args.type
        return 'rf_detr' if os.path.isfile(self.metrics_csv) else 'netharn'

    def name(self, run_type):
        return self.args.name or run_type

    def subject(self, run_type, text):
        return '[train-monitor:%s] %s %s' % (self.name(run_type), self.job_label, text)

    def running(self, log_text):
        if self.args.job_id:
            alive = slurm_running(self.args.job_id)
            if alive is not None:
                return alive
        if self.args.pid:
            alive = pid_running(self.args.pid)
            if alive is not None:
                return alive
        if self.args.done_pattern in log_text:
            return False
        return log_recent(self.log, self.args.stale_minutes)

    def elapsed(self):
        if self.args.job_id:
            text = slurm_elapsed(self.args.job_id)
            if text:
                return text
        secs = int(time.time() - self.started)
        return '%d:%02d:%02d since monitor start' % (secs // 3600, (secs // 60) % 60, secs % 60)

    # -- reporting -------------------------------------------------------
    def status_block(self, run_type, stage, log_text):
        out = [
            'job:       %s  (host %s, type %s)' % (self.job_label, self.host, run_type),
            'elapsed:   %s' % self.elapsed(),
            'stage:     %s' % stage,
        ]
        if run_type == 'rf_detr':
            out.append('max step:  %s' % latest_step(self.metrics_csv))
        if self.args.chip_dir:
            out.append('chips:     %d' % count_files(self.args.chip_dir))
        out.append('disk free: %s' % disk_free(self.output_dir))
        if run_type == 'rf_detr':
            out.append('latest validation:')
            out.append(val_stats(self.metrics_csv, 'latest'))
        else:
            out.append('latest epoch line: %s' % (nh_latest_line(log_text) or '<none yet>'))
        return '\n'.join(out)

    def trail_line(self, run_type, stage, log_text, err):
        free = disk_free(self.output_dir)
        suffix = ' | ERROR: %s' % err if err else ''
        if run_type == 'rf_detr':
            return 'step=%s | %s | disk_free=%s%s' % (
                latest_step(self.metrics_csv), latest_best_map(log_text), free, suffix)
        chips = ''
        if self.args.chip_dir:
            chips = 'chips=%d | ' % count_files(self.args.chip_dir)
        return 'stage=%s | %s%s | disk_free=%s%s' % (
            stage, chips, nh_latest_line(log_text), free, suffix)

    def end_report(self, run_type, log_text):
        stage = nh_stage(log_text, self.args.done_pattern)
        # A swallowed error (e.g. CUDA OOM) can still print the done pattern,
        # so an error in the log takes precedence over a "finished" verdict.
        err = last_error(log_text)
        if err:
            subject, reason = 'ENDED with ERROR', 'ENDED_WITH_ERROR: %s' % err
        elif self.args.done_pattern in log_text:
            subject, reason = 'FINISHED normally', 'COMPLETED_OK'
        else:
            subject, reason = 'ENDED unexpectedly', 'ENDED_UNEXPECTEDLY'
        append_status(self.status_log, 'JOB NO LONGER RUNNING: %s' % reason)
        tail = [ln for ln in tail_lines(self.log, 25) if not NOISE_RE.search(ln)]
        self.notifier.send(
            self.subject(run_type, subject),
            'reason: %s\n\n%s\n\nlast log lines:\n%s' % (
                reason, self.status_block(run_type, stage, log_text), '\n'.join(tail)))

    # -- main loop -------------------------------------------------------
    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        log_text = self.read_log()
        run_type = self.resolve_type()
        stage = nh_stage(log_text, self.args.done_pattern)
        append_status(self.status_log, 'monitor started for %s (type %s); %s' % (
            self.job_label, run_type, self.notifier.describe()))
        self.notifier.send(
            self.subject(run_type, 'started (test report)'),
            'This confirms monitor reports are working.\n\n%s' % (
                self.status_block(run_type, stage, log_text)))

        emailed_error = False
        last_reported_epoch = -1
        last_stage = stage
        polls_since_beat = 0
        interval = max(1, self.args.epoch_interval)

        while True:
            log_text = self.read_log()
            run_type = self.resolve_type()
            if not self.running(log_text):
                # Give the run's final output a moment to land, then re-read
                # so the verdict sees its last lines.
                time.sleep(5)
                self.end_report(run_type, self.read_log())
                return 0

            stage = nh_stage(log_text, self.args.done_pattern)
            err = last_error(log_text)
            append_status(self.status_log, self.trail_line(run_type, stage, log_text, err))

            if err and not emailed_error:
                self.notifier.send(
                    self.subject(run_type, 'ERROR detected'),
                    'A possible error/deadlock was detected:\n%s\n\n%s' % (
                        err, self.status_block(run_type, stage, log_text)))
                emailed_error = True

            if stage != last_stage:
                self.notifier.send(
                    self.subject(run_type, '- stage: %s' % stage),
                    'Training stage changed: %s -> %s\n\n%s' % (
                        last_stage, stage, self.status_block(run_type, stage, log_text)))
                last_stage = stage
                polls_since_beat = 0

            if run_type == 'rf_detr':
                # Every N epochs (1-indexed: epochs N-1, 2N-1, ...).
                epoch = latest_val_epoch(self.metrics_csv)
                if epoch >= 0 and epoch != last_reported_epoch \
                        and (epoch + 1) % interval == 0:
                    self.notifier.send(
                        self.subject(run_type, '- validation at epoch %d' % epoch),
                        '%s\n\n%s' % (val_stats(self.metrics_csv, epoch),
                                      self.status_block(run_type, stage, log_text)))
                    last_reported_epoch = epoch
            else:
                # Per-epoch report at each interval, else a periodic heartbeat
                # so status still arrives during long stages.
                epoch = nh_epoch(log_text)
                polls_since_beat += 1
                if epoch >= 0 and epoch != last_reported_epoch and epoch % interval == 0:
                    self.notifier.send(self.subject(run_type, '- epoch %d' % epoch),
                                       self.status_block(run_type, stage, log_text))
                    last_reported_epoch = epoch
                    polls_since_beat = 0
                elif polls_since_beat >= self.args.heartbeat_polls:
                    self.notifier.send(self.subject(run_type, '- heartbeat (%s)' % stage),
                                       self.status_block(run_type, stage, log_text))
                    polls_since_beat = 0

            time.sleep(self.args.poll_seconds)


# ----------------------------------------------------------------- start/stop --

def settings_path(output_dir):
    return os.path.join(output_dir, SETTINGS_FILE)


def pid_path(output_dir):
    return os.path.join(output_dir, PID_FILE)


def read_pid(output_dir):
    try:
        with open(pid_path(output_dir)) as fin:
            return int(fin.read().strip())
    except (OSError, ValueError):
        return None


def process_identity(pid):
    """Creation identity prevents a stale PID file from targeting a reused PID."""
    try:
        if sys.platform.startswith('linux'):
            with open('/proc/%d/stat' % pid) as stream:
                start = stream.read().rsplit(')', 1)[1].split()[19]
            with open('/proc/sys/kernel/random/boot_id') as stream:
                return stream.read().strip() + ':' + start
        if sys.platform == 'win32':
            cmd = ['powershell', '-NoProfile', '-Command',
                   '(Get-Process -Id %d).StartTime.ToUniversalTime().Ticks' % pid]
        else:
            cmd = ['ps', '-p', str(pid), '-o', 'lstart=']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def record_identity(output_dir, pid):
    with open(os.path.join(output_dir, 'monitor.identity.json'), 'w') as stream:
        json.dump({'pid': pid, 'identity': process_identity(pid)}, stream)


def owns_pid(output_dir, pid):
    try:
        with open(os.path.join(output_dir, 'monitor.identity.json')) as stream:
            record = json.load(stream)
        identity = process_identity(pid)
        return bool(identity and record.get('pid') == pid and record.get('identity') == identity)
    except (OSError, ValueError, AttributeError):
        return False


def detach(argv, output_dir):
    """Relaunch this script in the background and return the child pid."""
    out_path = os.path.join(output_dir, OUTPUT_FILE)
    out = open(out_path, 'a')
    kwargs = {'stdin': subprocess.DEVNULL, 'stdout': out, 'stderr': subprocess.STDOUT}
    if sys.platform == 'win32':
        kwargs['creationflags'] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs['start_new_session'] = True
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + argv, **kwargs)
    out.close()
    return proc.pid


def cmd_start(args, raw_argv):
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isfile(args.log):
        print('warning: log file %s does not exist yet' % args.log, file=sys.stderr)
    if args.foreground:
        with open(pid_path(output_dir), 'w') as fout:
            fout.write('%d\n' % os.getpid())
        record_identity(output_dir, os.getpid())
        try:
            return Monitor(args).run()
        except KeyboardInterrupt:
            return 130
        finally:
            try:
                os.remove(pid_path(output_dir))
            except OSError:
                pass

    existing = read_pid(output_dir)
    if existing and pid_running(existing):
        print('A monitor is already running for %s (pid %d); stop it first.'
              % (output_dir, existing), file=sys.stderr)
        return 1
    settings = {k: v for k, v in vars(args).items() if k not in ('func', 'foreground')}
    settings['output_dir'] = output_dir
    settings['log'] = os.path.abspath(args.log)
    settings['started'] = time.time()
    with open(settings_path(output_dir), 'w') as fout:
        json.dump(settings, fout, indent=2)
    pid = detach([a for a in raw_argv if a != '--foreground'] + ['--foreground'], output_dir)
    with open(pid_path(output_dir), 'w') as fout:
        fout.write('%d\n' % pid)
    record_identity(output_dir, pid)
    notifier = Notifier(args, os.path.join(output_dir, DEFAULT_STATUS_LOG))
    print('Monitor started (pid %d): %s' % (pid, notifier.describe()))
    print('Status trail: %s' % (args.status_log or os.path.join(output_dir, DEFAULT_STATUS_LOG)))
    print('Console output: %s' % os.path.join(output_dir, OUTPUT_FILE))
    return 0


def cmd_status(args, _raw_argv):
    output_dir = os.path.abspath(args.output_dir)
    pid = read_pid(output_dir)
    if pid and pid_running(pid):
        print('Monitor running (pid %d)' % pid)
    elif pid:
        print('Monitor pid %d recorded but not running' % pid)
    else:
        print('No monitor recorded in %s' % output_dir)

    try:
        with open(settings_path(output_dir)) as fin:
            settings = json.load(fin)
    except (OSError, ValueError):
        settings = None
    if settings:
        ns = argparse.Namespace(**settings)
        monitor = Monitor(ns)
        log_text = read_text(monitor.log)
        run_type = monitor.resolve_type()
        stage = nh_stage(log_text, ns.done_pattern)
        print('Run %s: %s' % ('alive' if monitor.running(log_text) else 'not running',
                              monitor.job_label))
        print(monitor.status_block(run_type, stage, log_text))
        status_log = monitor.status_log
    else:
        status_log = os.path.join(output_dir, DEFAULT_STATUS_LOG)
    if os.path.isfile(status_log):
        print('\nlast %d status lines (%s):' % (args.lines, status_log))
        for line in tail_lines(status_log, args.lines):
            print('  ' + line)
    return 0


def cmd_stop(args, _raw_argv):
    output_dir = os.path.abspath(args.output_dir)
    pid = read_pid(output_dir)
    if not pid:
        print('No monitor recorded in %s' % output_dir)
        return 1
    if not pid_running(pid):
        print('Monitor pid %d is not running' % pid)
    else:
        if not owns_pid(output_dir, pid):
            print('Refusing to stop pid %d: monitor process identity cannot be verified' % pid, file=sys.stderr)
            return 1
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/PID', str(pid), '/F'],
                               capture_output=True, timeout=30, check=True)
            else:
                os.kill(pid, signal.SIGTERM)
        except (OSError, subprocess.SubprocessError) as err:
            print('Could not stop pid %d: %s' % (pid, err), file=sys.stderr)
            return 1
        print('Stopped monitor pid %d' % pid)
    try:
        os.remove(pid_path(output_dir))
    except OSError:
        pass
    return 0


# -------------------------------------------------------------------- main ---

def build_parser():
    parser = argparse.ArgumentParser(
        prog='viame monitor',
        description='Monitor a training run and report progress by log or email.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:' + __doc__.split('Examples:', 1)[1])
    sub = parser.add_subparsers(dest='command', metavar='command')
    sub.required = True

    start = sub.add_parser('start', help='start monitoring a training run')
    start.add_argument('-o', '--output-dir', required=True,
                       help='training output/run folder (status trail lives here)')
    start.add_argument('-l', '--log', required=True,
                       help='training stdout/stderr log file')
    live = start.add_argument_group('run liveness (first available is used)')
    live.add_argument('--job-id', help='slurm job id to watch with squeue')
    live.add_argument('--pid', type=int, help='process id of the training run')
    live.add_argument('--stale-minutes', type=float, default=60,
                      help='without a job id or pid, treat the run as ended once the '
                           'log has not changed for this long (default 60)')
    start.add_argument('-t', '--type', choices=['auto', 'rf_detr', 'netharn'],
                       default='auto', help='training type (default auto)')
    start.add_argument('-n', '--name', help='label used in report subjects (default: type)')
    start.add_argument('--status-log', help='status trail file '
                       '(default OUTPUT_DIR/%s)' % DEFAULT_STATUS_LOG)
    start.add_argument('--epoch-interval', type=int, default=5,
                       help='report validation/epoch statistics every N epochs (default 5)')
    start.add_argument('--heartbeat-polls', type=int, default=6,
                       help='netharn: send a heartbeat every N polls (default 6)')
    start.add_argument('--chip-dir', help='netharn: chip folder to report progress from')
    start.add_argument('--poll-seconds', type=float, default=1200,
                       help='poll period in seconds (default 1200)')
    start.add_argument('--done-pattern', default=DEFAULT_DONE_PATTERN,
                       help='log substring meaning "finished normally" (default "%s")'
                            % DEFAULT_DONE_PATTERN)
    start.add_argument('--metrics-csv', help='rf_detr: metrics.csv path '
                       '(default OUTPUT_DIR/metrics.csv)')
    mail = start.add_argument_group('email reports')
    mail.add_argument('-e', '--email', help='recipient address (omit for trail-only reports)')
    mail.add_argument('--smtp-server', default=os.environ.get('VIAME_SMTP_SERVER') or None,
                      help='SMTP host[:port] (port 587 STARTTLS by default, 465 for SSL, '
                           '25 for plain; default from VIAME_SMTP_SERVER)')
    mail.add_argument('--smtp-user', default=os.environ.get('VIAME_SMTP_USER') or None,
                      help='SMTP login user (default from VIAME_SMTP_USER)')
    mail.add_argument('--smtp-password-env', default='VIAME_SMTP_PASSWORD',
                      help='environment variable holding the SMTP password '
                           '(default VIAME_SMTP_PASSWORD)')
    mail.add_argument('--smtp-from', default=os.environ.get('VIAME_SMTP_FROM') or None,
                      help='sender address (default from VIAME_SMTP_FROM, else '
                           'train-monitor@host)')
    mail.add_argument('--sendmail', help='sendmail binary when no SMTP server is given')
    start.add_argument('--foreground', action='store_true',
                       help='run in this terminal instead of detaching')
    start.set_defaults(func=cmd_start)

    status = sub.add_parser('status', help='show the current status of a monitored run')
    status.add_argument('output_dir', help='training output/run folder')
    status.add_argument('--lines', type=int, default=10,
                        help='status trail lines to show (default 10)')
    status.set_defaults(func=cmd_status)

    stop = sub.add_parser('stop', help='stop the monitor for a run')
    stop.add_argument('output_dir', help='training output/run folder')
    stop.set_defaults(func=cmd_stop)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'start':
        for name in ('poll_seconds', 'stale_minutes', 'epoch_interval', 'heartbeat_polls'):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0:
                parser.error(name.replace('_', '-') + ' must be finite and positive')
        if args.pid is not None and args.pid <= 0:
            parser.error('pid must be positive')
    return args.func(args, argv)


if __name__ == '__main__':
    sys.exit(main())
