# -*- coding: utf-8 -*-
"""
References:
    https://stackoverflow.com/questions/37300698/gaussian-mixture-model
    http://docs.opencv.org/trunk/db/d5c/tutorial_py_bg_subtraction.html

# Adjust the threshold value for left (thL) and right (thR) so code will
# select most fish without including non-fish objects (e.g. the net)
thL = 20
thR = 20

# Species classes
# This is fixed for each training set, so it will remain the same throughout an entire survey
# pollock, salmon unident., rockfish unident.
sp_numbs = [21740, 23202, 30040]

# Threshold for errors between before & after projected
# points to make matches between left and right
max_err = [50, 50]


# number to increment between frames
by_n = 1

# Factor to reduce the size of the image for processing
factor = 2
"""
from __future__ import division, print_function
import itertools as it
from collections import namedtuple
import numpy as np
import cv2

OrientedBBox = namedtuple('OrientedBBox', ('center', 'extent', 'angle'))


def get_rounded_scaled_dsize(dsize, scale):
    """
    Returns an integer size and scale that best approximates
    the floating point scale on the original size

    Args:
        dsize (tuple): original width height
        scale (float or tuple): desired floating point scale factor
    """
    try:
        sx, sy = scale
    except TypeError:
        sx = sy = scale
    w, h = dsize
    new_w = int(round(w * sx))
    new_h = int(round(h * sy))
    new_scale = new_w / w, new_h / h
    new_dsize = (new_w, new_h)
    return new_dsize, new_scale


def imscale(img, scale):
    """
    Resizes an image by a scale factor (returns the rounded scale factor)
    """
    dsize = img.shape[0:2][::-1]
    new_dsize, new_scale = get_rounded_scaled_dsize(dsize, scale)
    new_img = cv2.resize(img, new_dsize, interpolation=cv2.INTER_LANCZOS4)
    return new_img, new_scale


def atleast_3channels(arr, copy=True):
    r"""
    Ensures that there are 3 channels in the image

    Args:
        arr (ndarray[N, M, ...]): the image
        copy (bool): Always copies if True, if False, then copies only when the
            size of the array must change.

    Returns:
        ndarray: with shape (N, M, C), where C in {3, 4}

    Doctest:
        >>> assert atleast_3channels(np.zeros((10, 10))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 1))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 3))).shape[-1] == 3
        >>> assert atleast_3channels(np.zeros((10, 10, 4))).shape[-1] == 4
    """
    ndims = len(arr.shape)
    if ndims == 2:
        res = np.tile(arr[:, :, None], 3)
        return res
    elif ndims == 3:
        h, w, c = arr.shape
        if c == 1:
            res = np.tile(arr, 3)
        elif c in [3, 4]:
            res = arr.copy() if copy else arr
        else:
            raise ValueError('Cannot handle ndims={}'.format(ndims))
    else:
        raise ValueError('Cannot handle arr.shape={}'.format(arr.shape))
    return res


class FishDetector(object):
    def __init__(self):
        self.config = {
            # limits accepable targets to be within this region [padx, pady]
            'edge_limit': [12, 12],
            # Maximum aspect ratio for filtering out non-fish objects
            'max_aspect': 7.5,
            # Minimum aspect ratio for filtering out non-fish objects
            'min_aspect': 3.5,
            # minimum number of pixels to keep a section, after sections
            # are found by component function
            # 'min_size': 1500,
            'min_size': 2000,

            'factor': 2.0,

            'gmm.n_training_frames': 30,
            'gmm.initial_variance': 30,
        }

        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=self.config['gmm.n_training_frames'],
            varThreshold=self.config['gmm.initial_variance'] ** 2,
            detectShadows=False)

    def postprocess_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # opening is erosion followed by dilation
        mask = cv2.erode(src=mask, kernel=kernel, dst=mask)
        mask = cv2.dilate(src=mask, kernel=kernel, dst=mask)
        # do another dilation
        mask = cv2.dilate(src=mask, kernel=kernel, dst=mask)
        return mask

    def apply(self, img):

        if self.config['factor'] != 1.0:
            downfactor = 1 / self.config['factor']
            img, upfactor = imscale(img, downfactor)

        orig_mask = self.back_sub.apply(img)
        post_mask = self.postprocess_mask(orig_mask.copy())

        DRAWING = True

        # Mask for drawing
        if DRAWING:
            draw_mask = np.zeros(post_mask.shape[0:2], dtype=np.float)
            draw_mask[orig_mask > 0] = .2
            draw_mask[post_mask > 0] = .75

        detections = list(self.masked_detect(post_mask))

        # Initialize mask to store detections
        detect_mask = np.zeros(post_mask.shape, dtype=post_mask.dtype)
        for n, detection in enumerate(detections, start=1):
            detect_mask[detection['cc']] = n
            if DRAWING:
                draw_mask[detection['cc'] > 0] = 1.0

        if DRAWING:
            import vtool as vt
            from vtool.coverage_kpts import make_heatmask
            heat_mask = make_heatmask(draw_mask)
            alpha = heat_mask[:, :, 3].copy()
            # Adjust alpha to be more visible
            alpha[alpha > .5] = .9
            heat_mask[:, :, 3] = alpha
            draw_img = vt.blend_images(heat_mask, img, 'overlay')
            draw_img = (draw_img * 255).astype(np.uint8).copy()

        for detection in detections:
            # Points come back in (x, y), but we want to draw in (r, c)
            box_points = detection['box_points'].astype(np.int)
            hull_points = detection['hull'].astype(np.int)
            draw_img = cv2.drawContours(
                image=draw_img, contours=[hull_points], contourIdx=-1,
                color=(255, 0, 0), thickness=2)
            draw_img = cv2.drawContours(
                image=draw_img, contours=[box_points], contourIdx=-1,
                color=(0, 255, 0), thickness=2)

        masks = {
            'orig': orig_mask,
            'post': post_mask,
            'detect': detect_mask,
            'draw': draw_img,
        }

        n_detections = len(detections)
        print('n_detections = {!r}'.format(n_detections))
        return detections, masks

    def masked_detect(self, mask):
        """
        Find pixel locs of each cc and determine if its a valid detection
        """
        img_h, img_w = mask.shape

        # 4-way connected compoment algorithm
        n_ccs, cc_mask = cv2.connectedComponents(mask, connectivity=4)

        # Define thresholds to filter edges
        minx_lim, miny_lim = self.config['edge_limit']
        maxx_lim = img_w - minx_lim
        maxy_lim = img_h - miny_lim

        # Return only components that satisfy certain properties
        for cc_label in range(1, n_ccs):
            cc = (cc_mask == cc_label)
            # note, `np.where` returns coords in (r, c)
            cc_y, cc_x = np.where(cc)

            # Remove small regions
            n_pixels = len(cc_x)
            if n_pixels < self.config['min_size']:
                continue

            # Filter objects detected on the edge of the image region
            minx, maxx = cc_x.min(), cc_x.max()
            miny, maxy = cc_y.min(), cc_y.max()
            if any([minx < minx_lim, maxx > maxx_lim,
                    miny < miny_lim, maxy > maxy_lim]):
                continue

            # generate the valid detection
            points = np.vstack([cc_x, cc_y]).T

            hull = cv2.convexHull(points)
            oriented_bbox = OrientedBBox(*cv2.minAreaRect(hull))
            w, h = oriented_bbox.extent

            # Filter objects without fishy aspect ratios
            ar = max([(w / h), (h / w)])
            if any([ar < self.config['min_aspect'],
                    ar > self.config['max_aspect']]):
                continue

            detection = {
                'points': points,
                'oriented_bbox': oriented_bbox,
                'box_points': cv2.boxPoints(oriented_bbox),
                'cc': cc,
                'hull': hull[:, 0, :],
            }
            yield detection


def read_matlab_stereo_camera(cal_fpath):
    """
    References:
        http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html
    """
    import scipy.io
    cal_data = scipy.io.loadmat(cal_fpath)
    cal = cal_data['Cal']

    (om, T, fc_left, fc_right, cc_left, cc_right, kc_left, kc_right,
     alpha_c_left, alpha_c_right) = cal[0][0]
    stereo_camera = {
        'extrinsic': {
            'om': om.ravel(),  # rotation vector
            'T': T.ravel(),  # translation vector
        },
        'intrinsic_left': {
            'fc': fc_left.ravel(),  # focal point
            'cc': cc_left.ravel(),  # principle point
            'alpha_c': alpha_c_left.ravel()[0],  # skew
            'kc': kc_left.ravel(),  # distortion
        },
        'intrinsic_right': {
            'fc': fc_right.ravel(),
            'cc': cc_right.ravel(),
            'alpha_c': alpha_c_right.ravel()[0],
            'kc': kc_right.ravel(),
        },
    }
    return stereo_camera


class StereoCameras(object):
    """
    stereo = StereoCameras(cal)
    """
    def __init__(stereo, cal):
        stereo.cal = cal

    def camera_matrices(stereo):
        K1 = stereo.get_camera_matrix(stereo.cal['intrinsic_left'])
        K2 = stereo.get_camera_matrix(stereo.cal['intrinsic_right'])
        return K1, K2

    def distortions(stereo):
        kc1 = stereo.cal['intrinsic_left']['kc']
        kc2 = stereo.cal['intrinsic_right']['kc']
        return kc1, kc2

    def projection_matrices(stereo):
        """ Build the projection matrix (maps world-coord ==> pixels-coord) for
        each camera

        Notes:
            `P1 = K1 * [I3 | 0]`
            `P2 = K2 * [R12 | t12]`
        """
        K1 = stereo.get_camera_matrix(stereo.cal['intrinsic_left'])
        K2 = stereo.get_camera_matrix(stereo.cal['intrinsic_right'])
        P1 = K1.dot(np.hstack([np.eye(3), np.zeros((3, 1))]))

        R = np.diag(stereo.cal['extrinsic']['om'])
        T = stereo.cal['extrinsic']['T'][:, None]
        P2 = K2.dot(np.hstack([R, T]))
        return P1, P2

    def get_camera_matrix(stereo, intrin):
        fc = intrin['fc']
        cc = intrin['cc']
        alpha_c = intrin['alpha_c']
        KK = np.array([
            [fc[0], alpha_c * fc[0], cc[0]],
            [    0,           fc[1], cc[1]],
            [    0,               0,     1],
        ])
        return KK

    def componstate_distoration(stereo, pts_distort, kc):
        if len(kc) == 1:
            assert False
        else:
            k1, k2, k3, p1, p2 = kc

            # initial guess
            xd = pts_distort
            x = xd
            for _  in range(20):
                r_2 = (x ** 2).sum()
                k_radial =  1 + k1 * r_2 + k2 * r_2 ** 2 + k3 * r_2 ** 3

                delta_x = np.vstack([
                    2 * p1 * x.T[0] * x.T[1] + p2 * (r_2 + 2 * x.T[0] ** 2),
                    p1 * (r_2 + 2 * x.T[1] ** 2) + 2 * p2 * x.T[0] * x.T[1]
                ])
                x = (xd - delta_x.T) / k_radial
            return x

    def normalize_pixel(stereo, pts, key):
        intrinsic = stereo.cal['intrinsic_' + key]

        def take(d, keys):
            return [d[k] for k in keys]
        fc, cc, kc, alpha_c = take(intrinsic, ['fc', 'cc', 'kc', 'alpha_c'])

        xpts, ypts = pts.T

        # First: Subtract principal point, and divide by the focal length
        pts_distort = (pts - cc) / fc

        # Second: undo skew
        pts_distort.T[0] = pts_distort.T[0] - alpha_c * pts_distort.T[1]

        # Third: Compensate for lens distortion
        if np.linalg.norm(kc) != 0:
            pts_norm = stereo.componstate_distoration(pts_distort, kc)
        else:
            pts_norm = pts_distort
        return pts_norm

    def triangulate(stereo, pts1, pts2):
        """
        stereo = StereoCameras()
        """
        P1, P2 = stereo.projection_matrices()

        pts3d_homog = cv2.triangulatePoints(P1, P2, pts1, pts2)
        world_pts = (pts3d_homog[0:3] / pts3d_homog[3][None, :])
        return world_pts
        # np.linalg.norm(world_pts1 - world_pts2)

        # stereo.normalize_pixel(pts1, stereo.cal['intrinsic_left'])
        # stereo.normalize_pixel(pts1, stereo.cal['intrinsic_left'])


class FishStereo(object):
    def __init__(self):
        pass

    def stereo_triangulation():
        pass

    def find_match(self, detections1, detections2, cal, dsize):
        """
        """
        for det1, det2 in it.product(detections1, detections2):
            box_points1 = det1['box_points']
            box_points2 = det2['box_points']
            pts1 = box_points1[[0, 2]]
            pts2 = box_points2[[0, 2]]
            print('pts1 =\n{!r}'.format(pts1))
            print('pts2 =\n{!r}'.format(pts2))

            stereo = StereoCameras(cal)

            # pts = box_points1
            # intrinsic = cal['intrinsic_left']
            # Normalize the image projection according to the intrinsic parameters of the left and right cameras
            # pts1_norm = stereo.normalize_pixel(pts1, 'left')
            # pts2_norm = stereo.normalize_pixel(pts2, 'right')

            K1, K2 = stereo.camera_matrices()
            kc1, kc2 = stereo.distortions()

            R = np.diag(cal['extrinsic']['om'])
            T = cal['extrinsic']['T']

            rectified = cv2.stereoRectify(K1, kc1, K2, kc2, dsize, R, T)
            (R1, R2, P1, P2, Q, validPixROI1, validPixROI2) = rectified

            world_pts = stereo.triangulate(pts1, pts2)

            proj_pt_h = P1.dot(np.vstack([world_pts, [1] * len(world_pts.T)]))
            proj_pt = proj_pt_h[0:2] / proj_pt_h[2]
            print('pts1 =\n{!r}'.format(pts1))
            print('proj_pt =\n{!r}'.format(proj_pt))
            print('----')

            om = cal['extrinsic']['om']
            R = cv2.Rodrigues(om)

            np.linalg.norm(pts1[0] - pts1[1])
            np.linalg.norm(pts2[0] - pts2[1])


def demo():
    import glob
    from os.path import expanduser, join
    import cv2
    import ubelt as ub
    data_fpath = expanduser('~/data/autoprocess_test_set')
    cal_fpath = join(data_fpath, 'cal_201608.mat')

    cal = read_matlab_stereo_camera(cal_fpath)

    img_path1 = join(data_fpath, 'image_data/left')
    image_path_list1 = sorted(glob.glob(join(img_path1, '*.jpg')))

    img_path2 = join(data_fpath, 'image_data/right')
    image_path_list2 = sorted(glob.glob(join(img_path2, '*.jpg')))

    img_fpath1 = image_path_list1[0]

    dpath = ub.ensuredir('out')

    detector1 = FishDetector()
    detector2 = FishDetector()

    for frame_id, (img_fpath1, img_fpath2) in enumerate(zip(image_path_list1,
                                                            image_path_list2)):
        print('frame_id = {!r}'.format(frame_id))
        # step(i, img_fpath)
        img1 = cv2.imread(img_fpath1)
        img2 = cv2.imread(img_fpath2)

        detections1, masks1 = detector1.apply(img1)
        detections2, masks2 = detector2.apply(img2)

        import vtool as vt

        # stacked = vt.stack_images(masks1['draw'], masks2['draw'], vert=False)
        # cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_id), stacked)

        # stacked = vt.stack_images(masks1['orig'], masks2['orig'], vert=False)[0]
        # cv2.imwrite(dpath + '/mask{}_orig.png'.format(frame_id), stacked)

        stacked = vt.stack_images(masks1['draw'], masks2['draw'], vert=False)[0]
        cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_id), stacked)
        if len(detections1) > 0 and len(detections2) > 0:
            dsize = img1.shape[0:2]
            import utool
            utool.embed()
            FishStereo().find_match(detections1, detections2, cal, dsize)

if __name__ == '__main__':
    demo()
