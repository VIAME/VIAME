# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
import itertools

import torch
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
tqdm.monitor_interval = 0

from ...models import Siamese

from ..siamese_dataset import SiameseEXFDataLoader
from ..storage import DataStorage
from ..utilities import load_track_feature_file


def extract_siamese_features(model_path, data_storage, track_feature_file):
    # load trained model
    model = Siamese()
    model = torch.nn.DataParallel(model).cuda()

    snapshot = torch.load(model_path)
    model.load_state_dict(snapshot['state_dict'])
    print('Model loaded from {}'.format(model_path))
    model.train(False)

    # load data
    transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    kwargs = {'num_workers': 8, 'pin_memory': True}

    dids = load_track_feature_file(track_feature_file)[1]
    image_blobs = [data_storage.blob(did, 'img') for did in dids]

    data_loader_class = SiameseEXFDataLoader(image_blobs, transform=transform)
    test_loader = torch.utils.data.DataLoader(data_loader_class, batch_size=48, shuffle=False, **kwargs)

    pbar = tqdm(total=len(test_loader))

    idids = iter(dids)
    data_storage.close()
    for img in test_loader:
        input = Variable(img.cuda())

        with torch.no_grad():
            output = model(input)

        np_output = output.data.cpu().numpy()

        f_blobs = [data_storage.blob(did, 'app')
                   for did in itertools.islice(idids, len(np_output))]
        assert np_output.shape[0] == len(f_blobs)

        for npo, fb in zip(np_output, f_blobs):
            fb.write(npo.tobytes())

        pbar.update(1)


def main(model_path, data_root, train_feature_file, test_feature_file):
    with DataStorage(data_root) as data_storage:
        print("Extracting train features...")
        extract_siamese_features(model_path, data_storage, train_feature_file)
        print("Extracting test features...")
        extract_siamese_features(model_path, data_storage, test_feature_file)


def create_parser():
    p = argparse.ArgumentParser(description="Compute appearance features using a trained Siamese model")
    p.add_argument('--model-path', required=True,
                   help='Path to a trained Siamese model')
    p.add_argument('--data-root', required=True,
                   help='The path where all feature data is stored')
    p.add_argument('--train-feature-file', required=True,
                   help='Path to a track feature file for training data')
    p.add_argument('--test-feature-file', required=True,
                   help='Path to a track feature file for test data')
    return p


if __name__ == '__main__':
    main(**vars(create_parser().parse_args()))
