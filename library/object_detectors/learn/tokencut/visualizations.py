"""
Vis utilities. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import torch
import skimage.io
import numpy as np
from viame import image_kernels
import torch.nn as nn
from PIL import Image
import scipy

import matplotlib.pyplot as plt

def visualize_img(image, vis_folder, im_name):
    pltname = f"{vis_folder}/{im_name}"
    Image.fromarray(image).save(pltname)
    print(f"Original image saved at {pltname}.")

def visualize_predictions(img, pred, vis_folder, im_name, save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = np.copy(img)
    # Plot the box
    image_kernels.draw_rect(
        image,
        int(pred[0]), int(pred[1]),
        int(pred[2]) + 1, int(pred[3]) + 1,
        (255, 0, 0), 3,
    )
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_pred.jpg"
        Image.fromarray(image).save(pltname)
        print(f"Predictions saved at {pltname}.")
    return image
  
def visualize_predictions_gt(img, pred, gt, vis_folder, im_name, dim, scales, save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = np.copy(img)
    # Plot the box
    image_kernels.draw_rect(
        image,
        int(pred[0]), int(pred[1]),
        int(pred[2]) + 1, int(pred[3]) + 1,
        (255, 0, 0), 3,
    )
    # Plot the ground truth box
    if len(gt>1):
        for i in range(len(gt)):
            image_kernels.draw_rect(
                image,
                int(gt[i][0]), int(gt[i][1]),
                int(gt[i][2]) + 1, int(gt[i][3]) + 1,
                (0, 0, 255), 3,
            )
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_BBOX.jpg"
        Image.fromarray(image).save(pltname)
        #print(f"Predictions saved at {pltname}.")
    return image

def visualize_eigvec(eigvec, vis_folder, im_name, dim, scales, save=True):
    """
    Visualization of the second smallest eigvector
    """
    eigvec = scipy.ndimage.zoom(eigvec, scales, order=0, mode='nearest')
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_attn.jpg"
        plt.imsave(fname=pltname, arr=eigvec, cmap='cividis')
        print(f"Eigen attention saved at {pltname}.")
