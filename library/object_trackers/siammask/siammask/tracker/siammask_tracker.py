# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from viame import image_kernels
import numpy as np

from viame.object_trackers.siammask.siammask.core.config import cfg
from viame.object_trackers.siammask.siammask.utils.bbox import cxy_wh_2_rect
from viame.object_trackers.siammask.siammask.tracker.base_tracker import change, sz
from viame.object_trackers.siammask.siammask.tracker.siamrpn_tracker import SiamRPNTracker


class SiamMaskTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamMaskTracker, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTracker must have mask_head"
        assert hasattr(self.model, 'refine_head'), \
            "SiamMaskTracker must have refine_head"

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float32)
        crop = image_kernels.warp_affine(image, mapping, out_sz[0], out_sz[1],
                                         'bilinear', 'constant', padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THRESHOLD)
        target_mask = target_mask.astype(np.uint8)
        # No return-shape dance any more: OpenCV 2 and 3 returned
        # (image, contours, hierarchy) where 4 and 5 return two, and the
        # version test that picked between them was itself wrong on 5.x.
        contours = image_kernels.find_contours(target_mask)
        cnt_area = [image_kernels.contour_area(c) for c in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            polygon = contours[np.argmax(cnt_area)]
            # `min_area_rect` carries its own corners, so this is
            # cv2.boxPoints( cv2.minAreaRect( ... ) ) in one call
            rbox_in_img = image_kernels.min_area_rect(polygon)["corners"]
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x = round(s_x)

        x_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

        # Pass tracker's template features to shared model
        outputs = self.model.track(x_crop, self.zf)
        # Store intermediate features on tracker instance for mask refinement
        self.xf = outputs['xf']
        self.mask_corr_feature = outputs['mask_corr_feature']
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy,
                                                width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        # Pass tracker's stored features to mask refinement
        mask = self.model.mask_refine((delta_y, delta_x), self.xf, self.mask_corr_feature).sigmoid().squeeze()
        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()
        return {
                'bbox': bbox,
                'best_score': best_score,
                'mask': mask_in_img,
                'polygon': polygon,
               }
