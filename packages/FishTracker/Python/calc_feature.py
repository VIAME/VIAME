import cv2
import glob
import json
import logging
import numpy as np
import os
import sys

import googlenet

def decode_image(source):
    return cv2.imdecode(np.fromstring(source, np.uint8), cv2.IMREAD_COLOR)

def encode_image(image):
    success, destination = cv2.imencode('.jpg', image)
    assert success
    return destination.tostring()

def assert_image(source, size=None):
    """Asserts that the specified path is a valid image with exactly 3 (RGB) color channels."""
    image = decode_image(source)
    assert image is not None
    assert image.ndim == 3
    if size is not None:
        assert image.shape[:2] == size
    assert image.shape[2] == 3

def normalize_image(source, size=None):
    """Reads an image, performs basic validation, and saves a resized RGB JPEG version of the image."""
    try:
        image = decode_image(source)
        if image is None:
            raise IOError("Failed to read image")
        if image.ndim == 2:
            image = np.tile(image[:, :, None], (1, 1, 3))
        if size is not None:
            image = cv2.resize(image, size)
        destination = encode_image(image)
        assert_image(destination, size=size)
        return destination
    except:
        logging.exception("Failed to normalize image")
        return None

def main(argv):
    _, datadir = argv

    mapper = googlenet.make_mapper(os.environ['FS_ROOT'])

    data = dict()
    data['images'] = []
    for dirname in sorted(glob.glob(os.path.join(os.environ['FS_ROOT'], datadir, '*'))):
        paths =[]
        images = []
        for filename in sorted(glob.glob(os.path.join(os.environ['FS_ROOT'], datadir, dirname, '*'))):
            path = os.path.join(os.environ['FS_ROOT'], datadir, dirname, filename)
            print path
            paths.append(path)
            reader = open(path, 'rb')
            file = reader.read()
            reader.close()
            image = normalize_image(file, size=(224, 224))
            images.append(image)
        features = mapper(decode_image(image) for image in images)

        for i in range(len(paths)):
           image = dict()
           image['name'] = os.path.basename(paths[i])
           image['feature'] = features[i, :].tolist()
           data['images'].append(image)

    writer = open(os.path.join(os.environ['FS_ROOT'], os.path.dirname(datadir), 'features.json'), 'w')
    writer.write(json.dumps(data, indent = 4))
    writer.close()

if __name__ == '__main__':
    main(sys.argv)

