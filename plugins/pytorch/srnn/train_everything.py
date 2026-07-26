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
from pathlib import Path
import re
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


def visible_gpu_count():
    """Number of devices this process can see, at least one."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')

    if visible is not None:
        return max(len([d for d in visible.split(',') if d.strip()]), 1)

    try:
        import torch
        return max(torch.cuda.device_count(), 1)
    except Exception:
        return 1


def get_best_model(model_dir):
    """Return the epoch number and the path of the best-trained model in
    model_dir by validation loss.

    Model file names should have the form "snapshot_epoch_{epoch}.pt".

    """
    pattern = re.compile('Epoch ([^:]+): final vloss:([^ ]+)')
    vlosses = []
    with open(model_dir / 'log.txt') as f:
        for line in f:
            # Remove initial time stamp
            line = line[line.index(']') + 1:].lstrip()
            match = pattern.match(line)
            if match is not None:
                epoch, vloss = match.group(1, 2)
                assert int(epoch) == len(vlosses)
                vlosses.append(float(vloss))
    best_epoch = min(range(len(vlosses)), key=lambda ep: vlosses[ep])
    model = model_dir / 'snapshot_epoch_{}.pt'.format(best_epoch)
    assert model.exists()
    return best_epoch, model


def main(data_root, output_dir, stabilized, generate_options=None,
         lstm_model_params=None, lstm_train_options=None, tracks=None):
    output_dir.mkdir()
    print("Creating Siamese training data")
    gen_data = output_dir / 'training_data'
    gen_data_vids = gen_data / 'vids'
    gen_data_prefix = str(gen_data / 'out')
    # Run in process rather than shelling out, so the track states can be
    # handed over as objects. This stage was the only reader of the gt.kw18
    # files, which no longer exist.
    if tracks is None:
        raise ValueError(
            "Track states must be supplied; Siamese data generation no longer"
            " reads gt.kw18 files from data_root")

    from .generate_training_files_kw18 import generate_siamese_data

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
    run_mod(
        'siamese_main_train',
        model_dir=siamese_models,
        data_root=gen_data_vids,
        train_file=gen_data_prefix + '_siamese_train_set.p',
        test_file=gen_data_prefix + '_siamese_test_set.p',
    )

    best_epoch, _model = get_best_model(siamese_models)
    print("Selecting the epoch {} model".format(best_epoch))
    siamese_model = siamese_dir / 'best_model.pt'
    siamese_model.symlink_to(_model.relative_to(siamese_model.parent))

    print("Extracting appearance features")
    run_mod(
        'extract_siamese_features',
        model_path=siamese_model,
        data_root=gen_data_vids,
        train_feature_file=gen_data_prefix + '_train_features.p',
        test_feature_file=gen_data_prefix + '_test_features.p',
    )

    print("Creating LSTM training data")
    for fixed_length in (True, False):
        run_mod(
            'generate_training_files_kw18',
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

    n_gpus = visible_gpu_count()
    workers = min(n_gpus, len(jobs))

    print("Training {} individual LSTM models across {} device(s)"
          .format(len(jobs), workers))

    def train_one(index_and_job):
        index, (fixed_length, model_type) = index_and_job
        fix_letter = 'F' if fixed_length else 'V'
        name_key = model_type + '_' + fix_letter
        model_dir = lstm_dir / (name_key + '_models')
        gpu = index % n_gpus

        print("Training {} model on device {}".format(name_key, gpu))
        run_mod(
            'rnn_main_train',
            model_snapshot_dir=model_dir,
            data_root=gen_data_vids,
            train_file='_'.join([gen_data_prefix, fix_letter, 'train_set.p']),
            test_file='_'.join([gen_data_prefix, fix_letter, 'test_set.p']),
            RNN_Type=model_type[0].upper(),
            model_params=repr(lstm_model_params),
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
        best_epoch, _model = get_best_model(model_dir)
        print("Selecting the epoch {} model for {}".format(best_epoch, name_key))
        model = lstm_dir / (name_key + '_best.pt')
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
            **(lstm_train_options or {}),
        )
        best_epoch, _model = get_best_model(model_dir)
        print("Selecting the epoch {} model".format(best_epoch))
        model = target_lstm_dir / 'best_{}_model.pt'.format(fix_letter)
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
    # Not ideal
    p.add_argument('--generate-options', type=stringy_dict,
                   help='Extra options for generate_training_files_kw18.py'
                   ' as a Python dict literal')
    p.add_argument('--lstm-model-params', type=literal_eval,
                   help='Python dict literal with parameters for the LSTM model constructors')
    p.add_argument('--lstm-train-options', type=stringy_dict,
                   help='Extra options for rnn_main_train.py and target_rnn_main_train.py')
    return p


if __name__ == '__main__':
    main(**vars(create_parser().parse_args()))
