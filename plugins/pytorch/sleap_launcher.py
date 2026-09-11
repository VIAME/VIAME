# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See the root LICENSE file for details.
"""Train/export SLEAP-NN, or evaluate its VIAME model on an exported crop manifest.

Training: python -m viame.pytorch.sleap_launcher request.json
Testing:  python -m viame.pytorch.sleap_launcher request.json --evaluate model.pt
"""
import argparse
import json
from pathlib import Path
import numpy as np

from viame.pytorch.sleap_common import MODEL_FORMAT, SleapPredictor, load_artifact, parse_keypoint_names


def make_labels(records, names):
    import sleap_io as sio
    skeleton = sio.Skeleton(nodes=list(names))
    video = sio.Video.from_filename([row['image'] for row in records], grayscale=False)
    frames = [sio.LabeledFrame(video=video, frame_idx=i, instances=[
        sio.Instance.from_numpy(np.asarray(row['points'], dtype=float), skeleton=skeleton)
    ]) for i, row in enumerate(records)]
    return sio.Labels(labeled_frames=frames)


def make_training_config(options, train_path, val_path, output_dir):
    from sleap_nn.config.get_config import get_data_config, get_model_config, get_trainer_config
    from sleap_nn.config.training_job_config import TrainingJobConfig
    device = options['device']
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu' and device != 'cuda' and not device.startswith('cuda:'):
        raise ValueError('SLEAP training device must be auto, cpu, cuda, or cuda:N')
    indices = [int(device.split(':')[1])] if ':' in device else None
    names = parse_keypoint_names(options['keypoint_names'])
    return TrainingJobConfig(
        data_config=get_data_config(
            train_labels_path=[str(train_path)], val_labels_path=[str(val_path)],
            ensure_rgb=True, ensure_grayscale=False, scale=1.0,
            use_augmentations_train=options['augmentation'],
            intensity_aug=['contrast', 'brightness'] if options['augmentation'] else None,
            geometry_aug=dict(rotation_min=-180.0, rotation_max=180.0, rotation_p=0.5,
                              scale_min=0.9, scale_max=1.1, scale_p=0.5,
                              translate_width=0.05, translate_height=0.05, translate_p=0.5),
        ),
        model_config=get_model_config(
            backbone_config={'unet': dict(in_channels=3, filters=options['filters'],
                                         max_stride=options['max_stride'], output_stride=options['output_stride'])},
            head_configs={'single_instance': {'confmaps': dict(
                part_names=names, sigma=options['sigma'], output_stride=options['output_stride'])}},
        ),
        trainer_config=get_trainer_config(
            batch_size=options['batch_size'], num_workers=options['num_workers'],
            trainer_num_devices=1, trainer_device_indices=indices,
            trainer_accelerator='cpu' if device == 'cpu' else 'gpu',
            max_epochs=options['max_epochs'], seed=options['seed'],
            min_train_steps_per_epoch=1, train_steps_per_epoch=options['steps_per_epoch'] or None,
            learning_rate=options['learning_rate'], early_stopping=True,
            early_stopping_patience=options['patience'],
            ckpt_save_top_k=1, ckpt_save_last=True, save_ckpt=True,
            ckpt_dir=str(output_dir), run_name='training',
            use_wandb=False, visualize_preds_during_training=False,
        ),
        name=options['identifier'],
    ).to_sleap_nn_cfg()


def evaluate(model_path, records, device='cpu', batch_size=16):
    """Report missing predictions as failures in bbox-normalized PCK@0.05."""
    import cv2
    predictor = SleapPredictor(model_path, device=device)
    names = predictor.names
    errors = [[] for _ in names]
    labeled = np.zeros(len(names), dtype=int)
    predicted = np.zeros(len(names), dtype=int)
    correct = np.zeros(len(names), dtype=int)
    for start in range(0, len(records), batch_size):
        rows = records[start:start + batch_size]
        images = []
        for row in rows:
            image = cv2.imread(row['image'], cv2.IMREAD_COLOR)
            if image is None:
                raise OSError('Unable to read evaluation crop: %s' % row['image'])
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        points, _ = predictor.predict(images)
        for row, prediction in zip(rows, points):
            truth = np.asarray(row['points'], dtype=float)
            visible = np.isfinite(truth).all(axis=1)
            valid = visible & np.isfinite(prediction).all(axis=1)
            distance = np.linalg.norm(prediction - truth, axis=1)
            normalized = distance / row['box_diagonal']
            labeled += visible
            predicted += valid
            correct += valid & (normalized <= 0.05)
            for i in np.flatnonzero(valid):
                errors[i].append(float(normalized[i]))
    return dict(
        crops=len(records), keypoint_threshold=0.2,
        pck_normalization='unpadded bounding-box diagonal',
        keypoints={name: dict(labeled=int(labeled[i]), predicted=int(predicted[i]),
                             pck_005=float(correct[i] / labeled[i]) if labeled[i] else None,
                             mean_normalized_error=float(np.mean(errors[i])) if errors[i] else None)
                   for i, name in enumerate(names)},
    )


def run_training(request):
    import torch
    import sleap_nn
    import sleap_io as sio
    from omegaconf import OmegaConf
    from sleap_nn.training.model_trainer import ModelTrainer

    options = request['options']
    names = parse_keypoint_names(options['keypoint_names'])
    directory = Path(request['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    train_path, val_path = directory / 'train.slp', directory / 'val.slp'
    for split, path in (('train', train_path), ('val', val_path)):
        if not request['records'][split]:
            raise ValueError('No %s crops with visible keypoints' % split)
        sio.save_slp(make_labels(request['records'][split], names), str(path))
    config = make_training_config(options, train_path, val_path, directory)
    if options['seed_model']:
        seed = load_artifact(options['seed_model'])
        target_config = OmegaConf.to_container(config.model_config, resolve=True)
        if (seed['keypoint_names'] != names or
                seed['model_config']['backbone_config'] != target_config['backbone_config'] or
                seed['model_config']['head_configs'] != target_config['head_configs']):
            raise ValueError('seed_model must match the keypoint order and model architecture')
        native_seed = directory / 'seed.ckpt'
        torch.save({'state_dict': {'model.' + k: v for k, v in seed['state_dict'].items()}}, native_seed)
        config.model_config.pretrained_backbone_weights = str(native_seed)
        config.model_config.pretrained_head_weights = str(native_seed)
    trainer = ModelTrainer.get_model_trainer_from_config(config)
    trainer.train()
    best = trainer.trainer.checkpoint_callback.best_model_path
    if not best or not Path(best).is_file():
        raise RuntimeError('SLEAP finished without a best validation checkpoint')
    checkpoint = torch.load(best, map_location='cpu', weights_only=False)
    state = {key[len('model.'):]: value for key, value in checkpoint['state_dict'].items()
             if key.startswith('model.')}
    if not state:
        raise RuntimeError('SLEAP checkpoint does not contain model weights')
    model_config = OmegaConf.to_container(trainer.config.model_config, resolve=True)
    # Deployment must not refer back to training or seed paths.
    # SLEAP replaces part_names with Skeleton Node objects during setup.
    # Store portable strings so deployment needs no pickled SLEAP objects.
    model_config['head_configs']['single_instance']['confmaps']['part_names'] = names
    model_config['pretrained_backbone_weights'] = None
    model_config['pretrained_head_weights'] = None
    artifact = dict(format=MODEL_FORMAT, sleap_nn_version=sleap_nn.__version__,
                    keypoint_names=names, crop_size=[options['crop_height'], options['crop_width']],
                    crop_padding=options['crop_padding'], model_config=model_config, state_dict=state)
    output = directory / 'trained_keypoints.pt'
    torch.save(artifact, output)
    metrics = evaluate(output, request['records']['val'], options['device'], options['batch_size'])
    (directory / 'keypoint_metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--evaluate', type=Path, help='Evaluate an existing model instead of training')
    parser.add_argument('--output', type=Path, help='Evaluation report destination')
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if args.evaluate:
        artifact = load_artifact(args.evaluate)
        options = request['options']
        if (artifact['keypoint_names'] != parse_keypoint_names(options['keypoint_names']) or
                artifact['crop_size'] != [options['crop_height'], options['crop_width']] or
                artifact['crop_padding'] != options['crop_padding']):
            raise ValueError('Evaluation model must match the manifest keypoint order and crop settings')
        metrics = evaluate(args.evaluate, request['records']['val'],
                           request['options']['device'], request['options']['batch_size'])
        text = json.dumps(metrics, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(text)
        print(text)
    else:
        run_training(request)


if __name__ == '__main__':
    main()
