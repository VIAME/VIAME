# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import numpy as np
from viame import image_kernels

from util.utils import renorm
from util.misc import color_sys

_color_getter = color_sys(100)

# plot known and unknown box
def add_box_to_img(img, boxes, colorlist, brands=None):
    """[summary]

    Args:
        img ([type]): np.array, H,W,3
        boxes ([type]): list of list(4)
        colorlist: list of colors.
        brands: text.

    Return:
        img: np.array. H,W,3.
    """
    H, W = img.shape[:2]
    for _i, (box, color) in enumerate(zip(boxes, colorlist)):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        img = np.ascontiguousarray(img.copy())
        # `draw_rect`'s second corner is exclusive where cv2.rectangle's is
        # inclusive, hence the + 1 on each.
        image_kernels.draw_rect(img, int(x-w/2), int(y-h/2),
                                int(x+w/2) + 1, int(y+h/2) + 1, color, 2)
        if brands is not None:
            brand = brands[_i]
            # `draw_text` places by the **top left**, where cv2.putText
            # placed by the baseline, and draws a 5 by 7 bitmap font rather
            # than Hershey -- so this overlay is legible rather than
            # identical, which is what a debug overlay needs.
            image_kernels.draw_text(img, str(brand), int(x-w/2),
                                    int(y+h/2), color, 1)
    return img

def plot_dual_img(img, boxes, labels, idxs, probs=None):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor.
        boxes (): tensor(Kx4) or list of tensor(1x4).
        labels ([type]): list of ints.
        idxs ([type]): list of ints.
        probs (optional): listof floats.

    Returns:
        img_classcolor: np.array. H,W,3. img with class-wise label.
        img_seqcolor: np.array. H,W,3. img with seq-wise label.
    """
    # import ipdb; ipdb.set_trace()
    boxes = [i.cpu().tolist() for i in boxes]
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    # plot with class
    class_colors = [_color_getter(i) for i in labels]
    if probs is not None:
        brands = ["{},{:.2f}".format(j,k) for j,k in zip(labels, probs)]
    else:
        brands = labels
    img_classcolor = add_box_to_img(img, boxes, class_colors, brands=brands)
    # plot with seq
    seq_colors = [_color_getter((i * 11) % 100) for i in idxs]
    img_seqcolor = add_box_to_img(img, boxes, seq_colors, brands=idxs)
    return img_classcolor, img_seqcolor


def plot_raw_img(img, boxes, labels):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor. 
        boxes ([type]): Kx4. tensor
        labels ([type]): K. tensor.

    return:
        img: np.array. H,W,3. img with bbox annos.
    
    """
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    H, W = img.shape[:2]
    for box, label in zip(boxes.tolist(), labels.tolist()):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        # import ipdb; ipdb.set_trace()
        img = np.ascontiguousarray(img.copy())
        image_kernels.draw_rect(img, int(x-w/2), int(y-h/2),
                                int(x+w/2) + 1, int(y+h/2) + 1,
                                _color_getter(label), 2)
        # add text, placed by its top left rather than its baseline
        image_kernels.draw_text(img, str(label), int(x-w/2), int(y+h/2),
                                _color_getter(label), 2)

    return img