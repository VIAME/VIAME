# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Master orchestration script for SRNN tracker training.

This script runs the complete 6-step training pipeline:
1. Data preparation (Part 1) - Generate Siamese training data from KW18
2. Siamese model training - Train appearance feature extractor
3. Appearance feature extraction - Extract features using trained Siamese
4. Data preparation (Part 2) - Generate LSTM training data
5. Individual LSTM training - Train A/I/M/B LSTMs independently
6. Combined SRNN training - Train final TargetLSTM model

Usage:
    python -m viame.pytorch.srnn.train_everything data_root output_dir [options]

Example:
    python -m viame.pytorch.srnn.train_everything \\
        /path/to/training_data \\
        /path/to/output \\
        --stabilized
"""

import argparse
from ast import literal_eval
import os
from concurrent import futures
import pathlib
from pathlib import Path
import re
import shutil
import subprocess


def run(*args, **kwargs):
    """subprocess.run, but with check True by default"""
    if kwargs.get('check') is None:
        kwargs['check'] = True
    return subprocess.run(*args, **kwargs)


def run_mod_raw(args, *more_args, **kwargs):
    """run, but with the name of a module as the first argument"""
    args = ['python', '-m', 'viame.pytorch.srnn.' + args[0], *args[1:]]
    return run(args, *more_args, **kwargs)


def run_mod(name, *args, gpu=None, **kwargs):
    """Run a stage. gpu, when given, pins it to that visible device."""
    env = None

    if gpu is not None:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    run_mod_raw([name, *args, *(
        '--{}={}'.format(k.replace('_', '-'), v)
        for k, v in kwargs.items()
    )], env=env)


def visible_gpu_ids():
    """The devices this process can see, as CUDA_VISIBLE_DEVICES entries.

    A child pinned to one card is given one of these strings rather than its
    position in the list. CUDA_VISIBLE_DEVICES is read against the physical
    devices and not against the parent's already-restricted view, so a job
    allocated cards 4,5,6 that told a child "0" would send it to card 0, which
    is not one of its own.
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')

    if visible is not None:
        ids = [d.strip() for d in visible.split(',') if d.strip()]
        return ids or ['0']

    try:
        import torch
        return [str(d) for d in range(max(torch.cuda.device_count(), 1))]
    except Exception:
        return ['0']


def visible_gpu_count():
    """Number of devices this process can see, at least one."""
    return len(visible_gpu_ids())


def extract_worker_count():
    """How many processes to extract appearance features with.

    One per visible device by default. Extraction is a forward pass over every
    crop in the dataset -- around half a million of them on a full run, an hour
    of wall clock -- and running it in one process leaves the rest of a
    multi-card allocation idle for that hour.

    The override is for the cases the device count gets wrong: sharing cards
    with another job, or reducing to the undivided path deliberately.
    """
    override = os.environ.get('VIAME_SRNN_EXTRACT_WORKERS')

    if override:
        try:
            return max(int(override), 1)
        except ValueError:
            print("  ignoring unparseable VIAME_SRNN_EXTRACT_WORKERS={!r}"
                  .format(override))

    return visible_gpu_count()


def latest_sound_snapshot(model_dir):
    """The newest snapshot whose weights are all finite, and its epoch.

    A snapshot is written at the end of every epoch, including one that ended
    with a NaN loss, so the newest is not always a model worth carrying on
    from: resuming from NaN weights resumes nothing. Walk back until the
    weights are finite.

    Returns ( path, epoch ) or ( None, None ).
    """
    import re
    import torch

    if not model_dir.is_dir():
        return None, None

    snapshots = []

    for path in model_dir.glob('snapshot_epoch_*.pt'):
        match = re.search(r'snapshot_epoch_(\d+)\.pt$', path.name)

        if match:
            snapshots.append((int(match.group(1)), path))

    for epoch, path in sorted(snapshots, reverse=True):
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
            weights = state.get('state_dict', state)

            if all(torch.isfinite(t).all() for t in weights.values()
                   if torch.is_tensor(t) and t.is_floating_point()):
                return path, epoch

            print("  snapshot for epoch {} holds non-finite weights, "
                  "looking further back".format(epoch))
        except Exception as error:
            print("  could not read {}: {}".format(path.name, error))

    return None, None


def appearance_features_present(vids_dir):
    """Whether the appearance features have actually been written.

    They live in the detection store as blobs, not in a file, so the presence
    of a path says nothing about them. The _features.p files this stage is
    given are written by the data generation before it and exist either way:
    guarding on those skipped extraction on every resume, and the LSTM stage
    then failed unpacking an empty row for a feature that had never been
    stored.
    """
    import sqlite3

    db = pathlib.Path(vids_dir) / 'db.sqlite'

    if not db.exists():
        return False

    try:
        con = sqlite3.connect('file:{}?mode=ro'.format(db), uri=True)
    except sqlite3.Error:
        return False

    try:
        count = con.execute(
            "select count(*) from blobs where feature = 'app'").fetchone()[0]
        return count > 0
    except sqlite3.Error:
        return False
    finally:
        con.close()


def extract_appearance_features(model, vids_dir, feature_files,
                                loader_workers):
    """Push every crop through the trained Siamese model and store the result.

    The crops do not interact -- this is one forward pass each, in eval mode --
    so the work divides cleanly across the cards the job holds. Each worker
    takes a contiguous slice of the detection list, is pinned to one device,
    and writes its descriptors to a store of its own; the parent then folds
    those into the crop store in a single transaction.

    Separate stores are what makes this safe. Sqlite is happy with concurrent
    readers, which is all the workers are to the crop store, but concurrent
    writers to one file need care that nothing here would give them, and the
    crop store is the product of the longest stage in the pipeline. Only the
    merge writes to it, and only from this one process.

    The merge is also why a run killed partway through is recoverable. The
    resume guard asks whether appearance blobs are present at all, so a store
    left holding some of them would be taken for a finished stage and the LSTM
    training after it would fail unpacking a row that was never written. One
    transaction means the store holds all of them or none.
    """
    workers = extract_worker_count()

    # Nothing to divide and nothing to merge: run the stage exactly as the
    # single-device pipeline always has.
    if workers <= 1:
        run_mod(
            'extract_siamese_features',
            model_path=model,
            data_root=vids_dir,
            train_feature_file=feature_files[0],
            test_feature_file=feature_files[1],
            num_workers=loader_workers,
        )
        return

    from .storage import DataStorage
    from .utilities import load_track_feature_file

    expected = sum(len(load_track_feature_file(f)[1]) for f in feature_files)

    gpus = visible_gpu_ids()
    devices = [gpus[index % len(gpus)] for index in range(workers)]

    shard_root = Path(vids_dir).parent / 'feature_shards'
    shard_dirs = [shard_root / 'shard_{}'.format(index)
                  for index in range(workers)]

    # Shards left by an earlier attempt hold descriptors from a model that may
    # since have been retrained, and a shard from an interrupted worker holds
    # fewer than it should. Neither is worth keeping over recomputing it.
    shutil.rmtree(shard_root, ignore_errors=True)
    shard_root.mkdir(parents=True)

    print("  extracting {} feature(s) across {} worker(s) on device(s) {}"
          .format(expected, workers, ','.join(devices)))

    def extract_shard(index):
        run_mod(
            'extract_features_shard',
            model_path=model,
            data_root=vids_dir,
            train_feature_file=feature_files[0],
            test_feature_file=feature_files[1],
            shard_index=index,
            shard_count=workers,
            shard_root=shard_dirs[index],
            num_workers=loader_workers,
            gpu=devices[index],
        )

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # list() rather than leaving the iterator unconsumed: that is what
        # re-raises a failed worker's error here instead of losing it
        list(executor.map(extract_shard, range(workers)))

    print("  merging {} shard(s) into the detection store".format(workers))

    with DataStorage(str(vids_dir)) as data_storage:
        merged = data_storage.merge_feature(shard_dirs, 'app',
                                            expected_count=expected)

    print("  stored {} appearance feature(s)".format(merged))
    shutil.rmtree(shard_root, ignore_errors=True)


def get_best_model(model_dir):
    """Return the epoch number and the path of the best-trained model in
    model_dir by validation loss.

    Model file names should have the form "snapshot_epoch_{epoch}.pt".

    """
    pattern = re.compile('Epoch ([^:]+): final vloss:([^ ]+)')

    # Keyed by epoch rather than appended in order. A resumed stage writes its
    # epochs across more than one sitting and does not start from zero, so
    # requiring position to equal epoch number asserted on any resume. A later
    # record of the same epoch wins, being from the more recent run.
    vlosses = {}

    log = model_dir / 'log.txt'

    if log.exists():
        with open(log) as f:
            for line in f:
                try:
                    line = line[line.index(']') + 1:].lstrip()
                except ValueError:
                    continue

                match = pattern.match(line)

                if match is not None:
                    epoch, vloss = match.group(1, 2)

                    try:
                        vlosses[int(epoch)] = float(vloss)
                    except ValueError:
                        continue

    # Only consider epochs whose snapshot is actually there
    available = {epoch: loss for epoch, loss in vlosses.items()
                 if (model_dir / 'snapshot_epoch_{}.pt'.format(epoch)).exists()}

    if available:
        best_epoch = min(available, key=lambda ep: available[ep])
    else:
        # No usable record: keep the last epoch trained rather than failing.
        # Losing the log should cost the choice between snapshots, not the
        # stage and everything after it.
        snapshots = []

        for path in model_dir.glob('snapshot_epoch_*.pt'):
            match = re.search(r'snapshot_epoch_(\d+)\.pt$', path.name)

            if match:
                snapshots.append(int(match.group(1)))

        if not snapshots:
            raise RuntimeError(
                'No snapshots in {} to choose from'.format(model_dir))

        best_epoch = max(snapshots)
        print('  no validation losses recorded, keeping the last epoch '
              'trained ({})'.format(best_epoch))

    model = model_dir / 'snapshot_epoch_{}.pt'.format(best_epoch)
    assert model.exists()
    return best_epoch, model


def stage_done(*paths):
    """True if every path exists, used to skip a stage already completed."""
    return all(Path(p).exists() for p in paths)


def main(data_root, output_dir, stabilized, generate_options=None,
         lstm_model_params=None, lstm_train_options=None, tracks=None,
         resume=False, lstm_concurrency=1, lstm_loader_workers=2):
    """Run the SRNN pipeline.

    With resume set, a stage whose outputs are already present under
    output_dir is skipped. The pipeline has no checkpointing of its own and
    the early stages cost the bulk of the wall clock -- generating the
    training data and extracting features took about twenty of the twenty one
    hours of a full run here -- so being able to carry those over is what
    makes continuing on another machine worth doing.
    """
    output_dir.mkdir(exist_ok=resume)
    print("Creating Siamese training data")
    gen_data = output_dir / 'training_data'
    gen_data_vids = gen_data / 'vids'
    gen_data_prefix = str(gen_data / 'out')
    # Run in process rather than shelling out, so the track states can be
    # handed over as objects rather than round tripped through a file.
    siamese_sets = [gen_data_prefix + '_siamese_train_set.p',
                    gen_data_prefix + '_siamese_test_set.p']

    if resume and stage_done(*siamese_sets):
        print("  already generated, skipping")
    else:
        if tracks is None:
            raise ValueError(
                "Track states must be supplied; Siamese data generation does"
                " not read annotations from data_root")

        from .generate_training_files import generate_siamese_data

        generate_siamese_data(
            root_path=data_root,
            out_path=gen_data_vids,
            out_file_prefix=gen_data_prefix,
            tracks=tracks,
            stabilized=stabilized,
            **{k: v for k, v in (generate_options or {}).items()},
        )

    print("Training Siamese model")
    siamese_dir = output_dir / 'siamese'
    siamese_models = siamese_dir / 'models'
    siamese_model = siamese_dir / 'best_model.pt'

    if resume and stage_done(siamese_model):
        print("  already trained, skipping")
    else:
        siamese_options = {}

        # Carry on from the last sound epoch rather than starting the stage
        # again. It is the longest stage by far, and a run that dies partway
        # through has usually already paid for several epochs of it.
        if resume:
            snapshot, snapshot_epoch = latest_sound_snapshot(siamese_models)

            if snapshot is not None:
                print("  resuming from epoch {} ({})".format(
                    snapshot_epoch, snapshot.name))
                siamese_options['load_path'] = snapshot

        run_mod(
            'siamese_main_train',
            model_dir=siamese_models,
            data_root=gen_data_vids,
            train_file=gen_data_prefix + '_siamese_train_set.p',
            test_file=gen_data_prefix + '_siamese_test_set.p',
            num_workers=lstm_loader_workers,
            **siamese_options,
        )

        best_epoch, _model = get_best_model(siamese_models)
        print("Selecting the epoch {} model".format(best_epoch))
        siamese_model.unlink(missing_ok=True)
        siamese_model.symlink_to(_model.relative_to(siamese_model.parent))

    print("Extracting appearance features")
    feature_files = [gen_data_prefix + '_train_features.p',
                     gen_data_prefix + '_test_features.p']

    if resume and stage_done(*feature_files) \
            and appearance_features_present(gen_data_vids):
        print("  already extracted, skipping")
    else:
        extract_appearance_features(siamese_model, gen_data_vids,
                                    feature_files, lstm_loader_workers)

    print("Creating LSTM training data")
    for fixed_length in (True, False):
        fix_letter = 'F' if fixed_length else 'V'
        seq_sets = ['_'.join([gen_data_prefix, fix_letter, tt + '_set.p'])
                    for tt in ('train', 'test')]

        if resume and stage_done(*seq_sets):
            print("  {}-length already generated, skipping".format(fix_letter))
            continue

        run_mod(
            'generate_training_files',
            '--RNN-training',  # Well that's not ideal
            root_path=data_root,
            out_path=gen_data_vids,
            out_file_prefix=gen_data_prefix,
            fix_seq_flag='non-empty' if fixed_length else '',
            **(generate_options or {}),
        )

    # The eight individual LSTM trainings are independent of one another --
    # each writes its own snapshot directory and nothing reads another's output
    # until the combined stage below -- so they are run concurrently, one per
    # visible device, rather than eight times in series on a single card.
    model_types = ('app', 'motion', 'interaction', 'bbar')
    lstm_dir = output_dir / 'lstms'
    lstm_models = {}

    jobs = [
        (fixed_length, model_type)
        for fixed_length in (True, False)
        for model_type in model_types
    ]

    # Concurrency is capped separately from the device count. Each training
    # spawns its own data loading workers, and on Python 3.14 those come from a
    # forkserver rather than a fork, so every one is a fresh interpreter with
    # its own copy of the data. Running one per device overwhelmed a two device
    # node: the forkserver died and the trainings failed with BrokenPipeError.
    #
    # It may also go the other way, above the device count. These models are
    # small -- under a gigabyte of host memory and a few hundred megabytes of
    # video memory each, leaving a card that is mostly idle -- so several share
    # one comfortably, and gpu = index % n_gpus already round robins them. With
    # eight jobs and three devices the difference is one wave rather than three.
    # Unset still means one per device, which is what the machine that hit the
    # forkserver trouble wants.
    n_gpus = visible_gpu_count()
    workers = min(len(jobs), lstm_concurrency or n_gpus)

    print("Training {} individual LSTM models across {} device(s)"
          .format(len(jobs), workers))

    def train_one(index_and_job):
        index, (fixed_length, model_type) = index_and_job
        fix_letter = 'F' if fixed_length else 'V'
        name_key = model_type + '_' + fix_letter
        model_dir = lstm_dir / (name_key + '_models')
        # The entry from CUDA_VISIBLE_DEVICES, not the index. The value is
        # re-exported to the child outright, so an index would send a job
        # allocated cards 4,5,6 to physical card 0 -- the extraction stage
        # and the launcher script both pin this way for the same reason.
        gpu = visible_gpu_ids()[index % n_gpus]

        if resume and stage_done(lstm_dir / (name_key + '_best.pt')):
            print("Skipping {} model, already trained".format(name_key))
            return fixed_length, model_type, name_key, model_dir

        print("Training {} model on device {}".format(name_key, gpu))
        run_mod(
            'rnn_main_train',
            model_snapshot_dir=model_dir,
            data_root=gen_data_vids,
            train_file='_'.join([gen_data_prefix, fix_letter, 'train_set.p']),
            test_file='_'.join([gen_data_prefix, fix_letter, 'test_set.p']),
            RNN_Type=model_type[0].upper(),
            model_params=repr(lstm_model_params),
            num_workers=lstm_loader_workers,
            gpu=gpu,
            **(lstm_train_options or {}),
        )
        return fixed_length, model_type, name_key, model_dir

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(train_one, enumerate(jobs)))

    # Snapshot selection is bookkeeping, so it stays serial and ordered
    for fixed_length in (True, False):
        lstm_models['fixed' if fixed_length else 'var'] = {}

    for fixed_length, model_type, name_key, model_dir in results:
        model = lstm_dir / (name_key + '_best.pt')

        if not model.exists():
            best_epoch, _model = get_best_model(model_dir)
            print("Selecting the epoch {} model for {}".format(best_epoch,
                                                              name_key))
            # exists() is False for a dangling link but the name is still
            # taken, so symlink_to would fail on one left by an interrupted run
            model.unlink(missing_ok=True)
            model.symlink_to(_model.relative_to(model.parent))

        lstm_models['fixed' if fixed_length else 'var'][model_type] = model

    print("Training combined LSTM model")
    target_lstm_dir = output_dir / 'target_lstm'
    for fixed_length in (True, False):
        print("Training combined {}-length model"
              .format('fixed' if fixed_length else 'variable'))
        fix_letter = 'F' if fixed_length else 'V'
        source_models = lstm_models['fixed' if fixed_length else 'var']
        model_dir = target_lstm_dir / (fix_letter + '_models')
        model = target_lstm_dir / 'best_{}_model.pt'.format(fix_letter)

        if resume and stage_done(model):
            print("  already trained, skipping")
            continue

        run_mod(
            'target_rnn_main_train',
            model_dir=model_dir,
            **{k + '_load_path': v for k, v in source_models.items()},
            data_root=gen_data_vids,
            train_file='_'.join([gen_data_prefix, fix_letter, 'train_set.p']),
            test_file='_'.join([gen_data_prefix, fix_letter, 'test_set.p']),
            # XXX This should probably be customizable
            RNN_component='AIM',
            model_params=repr(lstm_model_params),
            num_workers=lstm_loader_workers,
            **(lstm_train_options or {}),
        )
        best_epoch, _model = get_best_model(model_dir)
        print("Selecting the epoch {} model".format(best_epoch))
        model.unlink(missing_ok=True)
        model.symlink_to(_model.relative_to(model.parent))


def stringy_dict(s):
    result = literal_eval(s)
    if not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in result.items()
    ):
        raise ValueError("Argument must be a string representation of a"
                         " Python dict literal with string keys and values")
    return result


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('data_root', type=Path,
                   help='Path to master training data folder')
    p.add_argument('output_dir', type=Path,
                   help='Path to organize all produced files under')
    p.add_argument('--stabilized', action='store_true',
                   help='Generate and train on stabilized data')
    p.add_argument('--resume', action='store_true',
                   help='Skip any stage whose outputs are already present'
                   ' under output_dir')
    # Not ideal
    p.add_argument('--generate-options', type=stringy_dict,
                   help='Extra options for generate_training_files.py'
                   ' as a Python dict literal')
    p.add_argument('--lstm-model-params', type=literal_eval,
                   help='Python dict literal with parameters for the LSTM model constructors')
    p.add_argument('--lstm-train-options', type=stringy_dict,
                   help='Extra options for rnn_main_train.py and target_rnn_main_train.py')
    return p


if __name__ == '__main__':
    main(**vars(create_parser().parse_args()))
