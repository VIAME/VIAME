# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
import sys
import os
import time

import torch.optim as optim

from torch.autograd import Variable
from random import randint

from viame.pytorch.mdnet.sample_generator import *
from viame.pytorch.mdnet.data_prov import *
from viame.pytorch.mdnet.model import *
from viame.pytorch.mdnet.bbreg import *
from viame.pytorch.mdnet.options import *
from viame.pytorch.mdnet.img_cropper import *

from roi_align import RoIAlignAvg
from roi_align import RoIAlignMax
from roi_align import RoIAlignAdaMax

def set_optimizer(model, lr_base,
                  lr_mult=opts['lr_mult'],
                  momentum=opts['momentum'],
                  w_decay=opts['w_decay']):

    params = model.get_learnable_params()
    param_list = []

    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})

    optimizer = optim.SGD(param_list,
                          lr = lr,
                          momentum=momentum,
                          weight_decay=w_decay)

    return optimizer


def train(model, criterion, optimizer, pos_feats,
          neg_feats, maxiter, in_layer='fc4'):

    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))

    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx,
          np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx,
          np.random.permutation(neg_feats.size(0))])

    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval() ## model transfer into evaluation mode
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train() ## model transfer into train mode

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        if opts['visual_log'] and len(loss.size()) != 0:
            print("Iter %d, Loss %.4f" % (iter, loss.data[0]))


class MDNetTracker:
    def __init__(self, first_image, init_bbox):

        # Init last bbox
        self._target_bbox = np.array(init_bbox)

        # Track execution times and counters for update planning
        self._exec_times = []
        self._frame_counter = 0

        # Initialize model
        if isinstance(opts['model_seed'], str):
            self._model = MDNet(opts['model_seed'])
        else:
            self._model = type(opts['model_seed'])()
            self._model.load_state_dict(opts['model_seed'].state_dict())

        if opts['adaptive_align']:
            align_h = self._model.roi_align_model.aligned_height
            align_w = self._model.roi_align_model.aligned_width
            spatial_s = self._model.roi_align_model.spatial_scale
            self._model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
        if opts['use_gpu']:
            self._model = self._model.cuda()

        self._model.set_learnable_params(opts['ft_layers'])

        # Init image cropper
        self._image_cropper = ImageCropper(1.)
        if opts['use_gpu']:
            self._image_cropper.gpuEnable()

        # Init criterion and optimizer
        self._criterion = BinaryLoss()
        init_optimizer = set_optimizer(self._model, opts['lr_init'])
        self._update_optimizer = set_optimizer(self._model, opts['lr_update'])
        tic = time.time()

        # Extract initial positive and negative samples
        ishape = first_image.shape
        pos_examples = gen_samples(SampleGenerator('gaussian',
          (ishape[1],ishape[0]), 0.1, 1.2),
          self._target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
        neg_examples = gen_samples(SampleGenerator('uniform',
          (ishape[1],ishape[0]), 1, 2, 1.1),
          self._target_bbox, opts['n_neg_init'], opts['overlap_neg_init'])
        neg_examples = np.random.permutation(neg_examples)

        cur_bbreg_examples = gen_samples(SampleGenerator('uniform',
          (ishape[1],ishape[0]), 0.3, 1.5, 1.1),
          self._target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])

        # Compute padded sample
        padded_x1 = (neg_examples[:,0]-neg_examples[:,2]*(opts['padding']-1.)/2.).min()
        padded_y1 = (neg_examples[:,1]-neg_examples[:,3]*(opts['padding']-1.)/2.).min()
        padded_x2 = (neg_examples[:,0]+neg_examples[:,2]*(opts['padding']+1.)/2.).max()
        padded_y2 = (neg_examples[:,1]+neg_examples[:,3]*(opts['padding']+1.)/2.).max()

        padded_scene_box = np.reshape(np.asarray(
          (padded_x1,padded_y1,padded_x2-padded_x1,padded_y2-padded_y1)),(1,4))

        scene_boxes = np.reshape(np.copy(padded_scene_box), (1,4))

        if opts['jitter']:
            ## horizontal shift
            jittered_scene_box_horizon = np.copy(padded_scene_box)
            jittered_scene_box_horizon[0,0] -= 4.
            jitter_scale_horizon = 1.

            ## vertical shift
            jittered_scene_box_vertical = np.copy(padded_scene_box)
            jittered_scene_box_vertical[0,1] -= 4.
            jitter_scale_vertical = 1.

            jittered_scene_box_reduce1 = np.copy(padded_scene_box)
            jitter_scale_reduce1 = 1.1 ** (-1)

            ## vertical shift
            jittered_scene_box_enlarge1 = np.copy(padded_scene_box)
            jitter_scale_enlarge1 = 1.1 ** (1)

            ## scale reduction
            jittered_scene_box_reduce2 = np.copy(padded_scene_box)
            jitter_scale_reduce2 = 1.1**(-2)

            ## scale enlarge
            jittered_scene_box_enlarge2 = np.copy(padded_scene_box)
            jitter_scale_enlarge2 = 1.1 ** (2)

            scene_boxes = np.concatenate([scene_boxes, jittered_scene_box_horizon,
              jittered_scene_box_vertical,jittered_scene_box_reduce1,
              jittered_scene_box_enlarge1,jittered_scene_box_reduce2,
              jittered_scene_box_enlarge2],axis=0)

            jitter_scale = [1.,jitter_scale_horizon,jitter_scale_vertical,
              jitter_scale_reduce1,jitter_scale_enlarge1,
              jitter_scale_reduce2,jitter_scale_enlarge2]

        else:
            jitter_scale = [1.]

        self._model.eval()
        for bidx in range(0,scene_boxes.shape[0]):
            crop_img_size = (scene_boxes[bidx,2:4] * ((opts['crop_size'],opts['crop_size']) \
              / self._target_bbox[2:4])).astype('int64')*jitter_scale[bidx]
            cropped_image, first_image_var = self._image_cropper.crop_image(
              first_image, np.reshape(scene_boxes[bidx],(1,4)), crop_img_size)
            cropped_image = cropped_image - 128.0

            feat_map = self._model(cropped_image, out_layer='conv3')

            rel_target_bbox = np.copy(self._target_bbox)
            rel_target_bbox[0:2] -= scene_boxes[bidx,0:2]

            batch_num = np.zeros((pos_examples.shape[0], 1))
            cur_pos_rois = np.copy(pos_examples)
            cur_pos_rois[:,0:2] -= np.repeat(np.reshape(
              scene_boxes[bidx,0:2],(1,2)),cur_pos_rois.shape[0],axis=0)
            scaled_obj_size = float(opts['crop_size'])*jitter_scale[bidx]
            cur_pos_rois = samples2maskroi(cur_pos_rois, self._model.receptive_field,
              (scaled_obj_size,scaled_obj_size), self._target_bbox[2:4], opts['padding'])
            cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
            cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
            cur_pos_feats = self._model.roi_align_model(feat_map, cur_pos_rois)
            cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

            batch_num = np.zeros((neg_examples.shape[0], 1))
            cur_neg_rois = np.copy(neg_examples)
            cur_neg_rois[:,0:2] -= np.repeat(np.reshape(scene_boxes[bidx,0:2],(1,2)),
              cur_neg_rois.shape[0],axis=0)
            cur_neg_rois = samples2maskroi(cur_neg_rois, self._model.receptive_field,
              (scaled_obj_size,scaled_obj_size), self._target_bbox[2:4], opts['padding'])
            cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
            cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
            cur_neg_feats = self._model.roi_align_model(feat_map, cur_neg_rois)
            cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

            ## bbreg rois
            batch_num = np.zeros((cur_bbreg_examples.shape[0], 1))
            cur_bbreg_rois = np.copy(cur_bbreg_examples)
            cur_bbreg_rois[:,0:2] -= np.repeat(np.reshape(
              scene_boxes[bidx,0:2],(1,2)),cur_bbreg_rois.shape[0],axis=0)
            scaled_obj_size = float(opts['crop_size'])*jitter_scale[bidx]
            cur_bbreg_rois = samples2maskroi(cur_bbreg_rois, self._model.receptive_field,
              (scaled_obj_size,scaled_obj_size), self._target_bbox[2:4], opts['padding'])
            cur_bbreg_rois = np.concatenate((batch_num, cur_bbreg_rois), axis=1)
            cur_bbreg_rois = Variable(torch.from_numpy(cur_bbreg_rois.astype('float32'))).cuda()
            cur_bbreg_feats = self._model.roi_align_model(feat_map, cur_bbreg_rois)
            cur_bbreg_feats = cur_bbreg_feats.view(cur_bbreg_feats.size(0), -1).data.clone()

            self._feat_dim = cur_pos_feats.size(-1)

            if bidx==0:
                pos_feats = cur_pos_feats
                neg_feats = cur_neg_feats

                ## bbreg feature
                bbreg_feats = cur_bbreg_feats
                bbreg_examples = cur_bbreg_examples
            else:
                pos_feats = torch.cat((pos_feats,cur_pos_feats),dim=0)
                neg_feats = torch.cat((neg_feats,cur_neg_feats),dim=0)

                ## bbreg feature
                bbreg_feats = torch.cat((bbreg_feats, cur_bbreg_feats),dim=0)
                bbreg_examples = np.concatenate((bbreg_examples, cur_bbreg_examples),axis=0)

        if pos_feats.size(0) > opts['n_pos_init']:
            pos_idx = np.asarray(range(pos_feats.size(0)))
            np.random.shuffle(pos_idx)
            pos_feats = pos_feats[pos_idx[0:opts['n_pos_init']],:]
        if neg_feats.size(0) > opts['n_neg_init']:
            neg_idx = np.asarray(range(neg_feats.size(0)))
            np.random.shuffle(neg_idx)
            neg_feats = neg_feats[neg_idx[0:opts['n_neg_init']], :]

        ## bbreg
        if bbreg_feats.size(0) > opts['n_bbreg']:
            bbreg_idx = np.asarray(range(bbreg_feats.size(0)))
            np.random.shuffle(bbreg_idx)
            bbreg_feats = bbreg_feats[bbreg_idx[0:opts['n_bbreg']], :]
            bbreg_examples = bbreg_examples[bbreg_idx[0:opts['n_bbreg']],:]

        ## open images and crop patch from obj
        extra_obj_size = np.array((opts['crop_size'],opts['crop_size']))
        extra_crop_img_size = extra_obj_size * (opts['padding']+0.6)
        replicateNum = 100
        for iidx in range(replicateNum):
            extra_target_bbox = np.copy(self._target_bbox)

            extra_scene_box = np.copy(extra_target_bbox)
            extra_scene_box_center = extra_scene_box[0:2] + extra_scene_box[2:4] / 2.
            extra_scene_box_size = extra_scene_box[2:4] * (opts['padding'] + 0.6)
            extra_scene_box[0:2] = extra_scene_box_center - extra_scene_box_size / 2.
            extra_scene_box[2:4] = extra_scene_box_size

            extra_shift_offset = np.clip(2. * np.random.randn(2), -4, 4)
            cur_extra_scale = 1.1 ** np.clip(np.random.randn(1), -2, 2)


            extra_scene_box[0] += extra_shift_offset[0]
            extra_scene_box[1] += extra_shift_offset[1]
            extra_scene_box[2:4] *= cur_extra_scale[0]

            scaled_obj_size = float(opts['crop_size']) / cur_extra_scale[0]

            cur_extra_cropped_image, _ = self._image_cropper.crop_image(first_image,
              np.reshape(extra_scene_box,(1,4)),extra_crop_img_size)
            cur_extra_cropped_image = cur_extra_cropped_image.detach()

            # extra_target_bbox = np.array(list(map(int, extra_target_bbox)))
            cur_extra_pos_examples = gen_samples(SampleGenerator('gaussian',
              (ishape[1], ishape[0]), 0.1, 1.2),extra_target_bbox,
              opts['n_pos_init']//replicateNum, opts['overlap_pos_init'])
            cur_extra_neg_examples = gen_samples(SampleGenerator('uniform',
              (ishape[1], ishape[0]), 0.3, 2, 1.1),extra_target_bbox,
              opts['n_neg_init']/replicateNum//4, opts['overlap_neg_init'])

            ## bbreg sample
            cur_extra_bbreg_examples = gen_samples(SampleGenerator('uniform',
              (ishape[1], ishape[0]), 0.3, 1.5, 1.1),
              extra_target_bbox, opts['n_bbreg']/replicateNum/4,
              opts['overlap_bbreg'], opts['scale_bbreg'])

            batch_num = iidx*np.ones((cur_extra_pos_examples.shape[0], 1))
            cur_extra_pos_rois = np.copy(cur_extra_pos_examples)
            cur_extra_pos_rois[:, 0:2] -= np.repeat(np.reshape(
              extra_scene_box[0:2], (1, 2)),
              cur_extra_pos_rois.shape[0], axis=0)
            cur_extra_pos_rois = samples2maskroi(cur_extra_pos_rois,
              self._model.receptive_field,(scaled_obj_size, scaled_obj_size),
              extra_target_bbox[2:4], opts['padding'])
            cur_extra_pos_rois = np.concatenate((batch_num,
              cur_extra_pos_rois), axis=1)

            batch_num = iidx * np.ones((cur_extra_neg_examples.shape[0], 1))
            cur_extra_neg_rois = np.copy(cur_extra_neg_examples)
            cur_extra_neg_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2],
              (1, 2)),cur_extra_neg_rois.shape[0], axis=0)
            cur_extra_neg_rois = samples2maskroi(cur_extra_neg_rois,
              self._model.receptive_field,(scaled_obj_size, scaled_obj_size),
              extra_target_bbox[2:4], opts['padding'])
            cur_extra_neg_rois = np.concatenate((batch_num, cur_extra_neg_rois), axis=1)

            ## bbreg rois
            batch_num = iidx * np.ones((cur_extra_bbreg_examples.shape[0], 1))
            cur_extra_bbreg_rois = np.copy(cur_extra_bbreg_examples)
            cur_extra_bbreg_rois[:,0:2] -= np.repeat(np.reshape(extra_scene_box[0:2],
              (1,2)),cur_extra_bbreg_rois.shape[0],axis=0)
            cur_extra_bbreg_rois = samples2maskroi(cur_extra_bbreg_rois,
              self._model.receptive_field,(scaled_obj_size,scaled_obj_size),
              extra_target_bbox[2:4], opts['padding'])
            cur_extra_bbreg_rois = np.concatenate((batch_num, cur_extra_bbreg_rois), axis=1)

            if iidx==0:
                extra_cropped_image = cur_extra_cropped_image

                extra_pos_rois = np.copy(cur_extra_pos_rois)
                extra_neg_rois = np.copy(cur_extra_neg_rois)

                ## bbreg rois
                extra_bbreg_rois = np.copy(cur_extra_bbreg_rois)
                extra_bbreg_examples = np.copy(cur_extra_bbreg_examples)
            else:
                extra_cropped_image = torch.cat((extra_cropped_image,
                  cur_extra_cropped_image),dim=0)

                extra_pos_rois = np.concatenate( (extra_pos_rois,
                  np.copy(cur_extra_pos_rois)), axis=0)
                extra_neg_rois = np.concatenate( (extra_neg_rois,
                  np.copy(cur_extra_neg_rois)), axis=0)

                ## bbreg rois
                extra_bbreg_rois = np.concatenate( (extra_bbreg_rois,
                  np.copy(cur_extra_bbreg_rois)), axis=0 )
                extra_bbreg_examples = np.concatenate( (extra_bbreg_examples,
                  np.copy(cur_extra_bbreg_examples)), axis=0 )

        extra_pos_rois = Variable(torch.from_numpy(extra_pos_rois.astype('float32'))).cuda()
        extra_neg_rois = Variable(torch.from_numpy(extra_neg_rois.astype('float32'))).cuda()

        ## bbreg rois
        extra_bbreg_rois = Variable(torch.from_numpy(extra_bbreg_rois.astype('float32'))).cuda()
        extra_cropped_image -= 128.

        extra_feat_maps = self._model(extra_cropped_image, out_layer='conv3')

        # Draw pos/neg samples
        ishape = first_image.shape

        extra_pos_feats = self._model.roi_align_model(extra_feat_maps, extra_pos_rois)
        extra_pos_feats = extra_pos_feats.view(extra_pos_feats.size(0), -1).data.clone()

        extra_neg_feats = self._model.roi_align_model(extra_feat_maps, extra_neg_rois)
        extra_neg_feats = extra_neg_feats.view(extra_neg_feats.size(0), -1).data.clone()

        ## BBreg feat
        extra_bbreg_feats = self._model.roi_align_model(extra_feat_maps, extra_bbreg_rois)
        extra_bbreg_feats = extra_bbreg_feats.view(extra_bbreg_feats.size(0), -1).data.clone()

        ## concatenate extra features to original_features
        pos_feats = torch.cat((pos_feats,extra_pos_feats),dim=0)
        neg_feats = torch.cat((neg_feats,extra_neg_feats), dim=0)

        ## concatenate extra bbreg feats to original_bbreg_feats
        bbreg_feats = torch.cat((bbreg_feats, extra_bbreg_feats), dim=0)
        bbreg_examples = np.concatenate((bbreg_examples, extra_bbreg_examples), axis=0)

        torch.cuda.empty_cache()
        self._model.zero_grad()

        # Initial training
        train(self._model, self._criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])

        ## BBReg train
        if bbreg_feats.size(0) > opts['n_bbreg']:
            bbreg_idx = np.asarray(range(bbreg_feats.size(0)))
            np.random.shuffle(bbreg_idx)
            bbreg_feats = bbreg_feats[bbreg_idx[0:opts['n_bbreg']],:]
            bbreg_examples = bbreg_examples[bbreg_idx[0:opts['n_bbreg']],:]
        self._bbreg = BBRegressor((ishape[1],ishape[0]))
        self._bbreg.train(bbreg_feats, bbreg_examples, self._target_bbox)

        if pos_feats.size(0) > opts['n_pos_update']:
            pos_idx = np.asarray(range(pos_feats.size(0)))
            np.random.shuffle(pos_idx)
            self._pos_feats_all = [pos_feats.index_select(0,
              torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())]
        if neg_feats.size(0) > opts['n_neg_update']:
            neg_idx = np.asarray(range(neg_feats.size(0)))
            np.random.shuffle(neg_idx)
            self._neg_feats_all = [neg_feats.index_select(0,
              torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())]

        self._spf_total = time.time()-tic
        #spf_total = 0. # no first frame

        # Display
        savefig = opts['savefig_dir'] != ''
        if opts['display'] or len(opts['savefig_dir']) > 0:
            dpi = 80.0
            figsize = (first_image.shape[1]/dpi, first_image.shape[0]/dpi)

            import matplotlib.pyplot as plt
            fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            im = ax.imshow(first_image)

            if gt is not None:
                gt_rect = plt.Rectangle(tuple(gt[0,:2]),gt[0,2],gt[0,3],
                        linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
                ax.add_patch(gt_rect)

            rect = plt.Rectangle(tuple(result_bb[0,:2]),result_bb[0,2],result_bb[0,3],
                    linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
            ax.add_patch(rect)

            if opts['display']:
                plt.pause(.01)
                plt.draw()
            if len(opts['savefig_dir']) > 0:
                fig.savefig(os.path.join(opts['savefig_dir'],'0000.jpg'),dpi=dpi)

    def update(self, image, likely_bbox=[]):
        trans_f = opts['trans_f']
        tic = time.time()
        self._frame_counter = self._frame_counter + 1

        # Estimate target bbox
        ishape = image.shape
        samples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]),
          trans_f, opts['scale_f'],valid=True), self._target_bbox, opts['n_samples'])

        padded_x1 = (samples[:, 0] - samples[:, 2]*(opts['padding']-1.)/2.).min()
        padded_y1 = (samples[:, 1] - samples[:,3]*(opts['padding']-1.)/2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2]*(opts['padding']+1.)/2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3]*(opts['padding']+1.)/2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1,
          padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > image.shape[1]:
            padded_scene_box[0] = image.shape[1]-1
        if padded_scene_box[1] > image.shape[0]:
            padded_scene_box[1] = image.shape[0]-1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0]+1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1]+1

        crop_img_size = (padded_scene_box[2:4] * ((opts['crop_size'], \
          opts['crop_size']) / self._target_bbox[2:4])).astype('int64')
        cropped_image,image_var = self._image_cropper.crop_image(image,
          np.reshape(padded_scene_box,(1,4)),crop_img_size)
        cropped_image = cropped_image - 128.

        self._model.eval()
        feat_map = self._model(cropped_image, out_layer='conv3')

        # relative target bbox with padded_scene_box
        rel_target_bbox = np.copy(self._target_bbox)
        rel_target_bbox[0:2] -= padded_scene_box[0:2]

        # Extract sample features and get target location
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2],
          (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois,self._model.receptive_field,
          (opts['crop_size'],opts['crop_size']), self._target_bbox[2:4],opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = self._model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = self._model(sample_feats, in_layer='fc4')
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.data.cpu().numpy()
        target_score = top_scores.data.mean()
        self._target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']

        # # Expand search area at failure
        if success:
            trans_f = opts['trans_f']
        else:
            trans_f = opts['trans_f_expand']

        ## Bbox regression
        if success:
            bbreg_feats = sample_feats[top_idx,:]
            bbreg_samples = samples[top_idx]
            bbreg_samples = self._bbreg.predict(bbreg_feats.data, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = self._target_bbox

        # Save result
        result = self._target_bbox
        result_bb = bbreg_bbox
        iou_result = 1.

        # Report computed, but use recommended box in training
        if len(likely_bbox) > 0:
            self._target_bbox = np.array(likely_bbox)

        # Data collect
        if success:

            # Draw pos/neg samples
            pos_examples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), self._target_bbox,
                opts['n_pos_update'],
                opts['overlap_pos_update'])
            neg_examples = gen_samples(
                SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), self._target_bbox,
                opts['n_neg_update'],
                opts['overlap_neg_update'])

            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()

            padded_scene_box = np.reshape(np.asarray((padded_x1, padded_y1,
              padded_x2 - padded_x1, padded_y2 - padded_y1)),(1,4))

            scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
            jitter_scale = [1.]

            for bidx in range(0, scene_boxes.shape[0]):
                crop_img_size = (scene_boxes[bidx, 2:4] * ((opts['crop_size'],
                  opts['crop_size']) / self._target_bbox[2:4])).astype('int64') * jitter_scale[bidx]
                cropped_image, image_var = self._image_cropper.crop_image(image,
                  np.reshape(scene_boxes[bidx], (1, 4)),crop_img_size)
                cropped_image = cropped_image - 128.

                feat_map = self._model(cropped_image, out_layer='conv3')

                rel_target_bbox = np.copy(self._target_bbox)
                rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

                batch_num = np.zeros((pos_examples.shape[0], 1))
                cur_pos_rois = np.copy(pos_examples)
                cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2],
                  (1, 2)), cur_pos_rois.shape[0],axis=0)
                scaled_obj_size = float(opts['crop_size']) * jitter_scale[bidx]
                cur_pos_rois = samples2maskroi(cur_pos_rois, self._model.receptive_field,
                  (scaled_obj_size, scaled_obj_size),self._target_bbox[2:4], opts['padding'])
                cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
                cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
                cur_pos_feats = self._model.roi_align_model(feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

                batch_num = np.zeros((neg_examples.shape[0], 1))
                cur_neg_rois = np.copy(neg_examples)
                cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2],
                  (1, 2)), cur_neg_rois.shape[0],
                                                  axis=0)
                cur_neg_rois = samples2maskroi(cur_neg_rois, self._model.receptive_field,
                  (scaled_obj_size, scaled_obj_size),
                  self._target_bbox[2:4], opts['padding'])
                cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
                cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
                cur_neg_feats = self._model.roi_align_model(feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

                self._feat_dim = cur_pos_feats.size(-1)

                if bidx == 0:
                    pos_feats = cur_pos_feats ##index select
                    neg_feats = cur_neg_feats
                else:
                    pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                    neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)

            if pos_feats.size(0) > opts['n_pos_update']:
                pos_idx = np.asarray(range(pos_feats.size(0)))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats.index_select(0,
                  torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())
            if neg_feats.size(0) > opts['n_neg_update']:
                neg_idx = np.asarray(range(neg_feats.size(0)))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats.index_select(0,
                  torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())

            self._pos_feats_all.append(pos_feats)
            self._neg_feats_all.append(neg_feats)

            if len(self._pos_feats_all) > opts['n_frames_long']:
                del self._pos_feats_all[0]
            if len(self._neg_feats_all) > opts['n_frames_short']:
                del self._neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'],len(self._pos_feats_all))
            pos_data = torch.stack(self._pos_feats_all[-nframes:],0).view(-1,self._feat_dim)
            neg_data = torch.stack(self._neg_feats_all,0).view(-1,self._feat_dim)
            train(self._model, self._criterion, self._update_optimizer,
              pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif self._frame_counter % opts['long_interval'] == 0:
            pos_data = torch.stack(self._pos_feats_all,0).view(-1,self._feat_dim)
            neg_data = torch.stack(self._neg_feats_all,0).view(-1,self._feat_dim)
            train(self._model, self._criterion, self._update_optimizer,
              pos_data, neg_data, opts['maxiter_update'])

        spf = time.time()-tic
        self._spf_total += spf

        # Display
        if opts['display'] or len(opts['savefig_dir']) > 0:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i,:2])
                gt_rect.set_width(gt[i,2])
                gt_rect.set_height(gt[i,3])

            rect.set_xy(result_bb[:2])
            rect.set_width(result_bb[2])
            rect.set_height(result_bb[3])

            if opts['display']:
                import matplotlib.pyplot as plt
                plt.pause(.01)
                plt.draw()
            if len(opts['savefig_dir']) > 0:
                fig.savefig(os.path.join(opts['savefig_dir'],'%04d.jpg'%(i)),dpi=dpi)

        if opts['visual_log']:
            print("Frame %d, Score %.3f, Time %.3f" % \
                (self._frame_counter, target_score, spf))

        return result_bb, target_score
  
