# -*- coding: utf-8 -*-
"""
Reimplementation of matlab algorithms for fishlength detection in python.
This is the first step of moving them into kwiver.

# Adjust the threshold value for left (thL) and right (thR) so code will
# select most fish without including non-fish objects (e.g. the net)
thL = 20
thR = 20

# Species classes
# This is fixed for each training set, so it will remain the same throughout an entire survey
# pollock, salmon unident., rockfish unident.
sp_numbs = [21740, 23202, 30040]


# number to increment between frames
by_n = 1

# Factor to reduce the size of the image for processing
factor = 2
"""
from __future__ import division, print_function
from collections import namedtuple
import cv2
import itertools as it
import numpy as np
import scipy.optimize
from imutils import ( # NOQA
    imscale, ensure_grayscale, overlay_heatmask, from_homog, to_homog,
    downsample_average_blocks)
from os.path import expanduser, basename, join

try:
    import utool as ut
    print, rrr, profile = ut.inject2(__name__)
except ImportError:
    def profile(func):
        return func


OrientedBBox = namedtuple('OrientedBBox', ('center', 'extent', 'angle'))


def grabcut(bgr_img, prior_mask, binary=True, num_iters=5):
    """
    Referencs:
        http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html
    """
    # Grab Cut Parameters
    (h, w) = bgr_img.shape[0:2]
    rect = (0, 0, w, h)

    mode = cv2.GC_INIT_WITH_MASK
    bgd_model = np.zeros((1, 13 * 5), np.float64)
    fgd_model = np.zeros((1, 13 * 5), np.float64)
    # Grab Cut Execution
    post_mask = prior_mask.copy()
    if binary:
        is_pr_bgd = (post_mask == 0)
        if np.all(is_pr_bgd) or not np.any(is_pr_bgd):
            return post_mask
        post_mask[post_mask > 0]  = cv2.GC_FGD
        post_mask[post_mask == 0] = cv2.GC_PR_BGD

    cv2.grabCut(bgr_img, post_mask, rect, bgd_model, fgd_model, num_iters, mode=mode)
    if binary:
        is_forground = (post_mask == cv2.GC_FGD) + (post_mask == cv2.GC_PR_FGD)
        post_mask = np.where(is_forground, 255, 0).astype('uint8')
    else:
        label_colors = [       255,           170,            50,          0]
        label_values = [cv2.GC_FGD, cv2.GC_PR_FGD, cv2.GC_PR_BGD, cv2.GC_BGD]
        pos_list = [post_mask == value for value in label_values]
        for pos, color in zip(pos_list, label_colors):
            post_mask[pos] = color
    return post_mask


class FishDetector(object):
    """
    Uses background subtraction and 4-way connected compoments algorithm to
    detect potential fish objects. Objects are filtered by size, aspect ratio,
    and closeness to the image border to remove bad detections.

    References:
        https://stackoverflow.com/questions/37300698/gaussian-mixture-model
        http://docs.opencv.org/trunk/db/d5c/tutorial_py_bg_subtraction.html

    """
    def __init__(self, **kwargs):
        bg_algo = kwargs.get('bg_algo', 'gmm')

        self.config = {
            # limits accepable targets to be within this region [padx, pady]
            # These are wrt the original image size
            'edge_trim': [12, 12],

            # Min/Max aspect ratio for filtering out non-fish objects
            'aspect_thresh': (3.5, 7.5,),

            # are found by component function
            'bg_algo': bg_algo,
        }

        self.n_iters = 0

        # Different default params depending on the background subtraction algo
        if self.config['bg_algo'] == 'median':
            self.config.update({
                'factor': 4.0,
                # minimum number of pixels to keep a section, wrt original size
                'min_size': 800,
                'diff_thresh': 19,
                'smooth_ksize': None,
                'local_threshold': True,
            })
        else:
            self.config.update({
                'factor': 2.0,
                'min_size': 100,
                'n_training_frames': 30,
                'gmm_thresh': 30,
                'smooth_ksize': (10, 10),  # wrt original image size
                'local_threshold': False,
            })

        self.config.update(kwargs)

        # Choose which algo to use for background subtraction
        if self.config['bg_algo'] == 'median':
            self.background_model = MedianBackgroundSubtractor(
                diff_thresh=self.config['diff_thresh'],
            )
        elif self.config['bg_algo'] == 'gmm':
            self.background_model = cv2.createBackgroundSubtractorMOG2(
                history=self.config['n_training_frames'],
                varThreshold=self.config['gmm_thresh'],
                detectShadows=False)
        # self.background_model = cv2.createBackgroundSubtractorKNN(
        #     history=self.config['n_training_frames'],
        #     dist2Threshold=50 ** 2,
        #     detectShadows=False
        # )

    @profile
    def apply(self, img, return_info=True):
        """
        Detects the objects in the image and update the background model.

        Args:
            img (ndarray): image to perform detection on
            return_info (bool): returns intermediate plotting data if True

        Returns:
            detections, masks : list of dicts, dict of ndarrays
                detection dicts and a dict of intermediate processing stages
                for algorithm visualization.

        Doctest:
            % pylab qt5
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> self, img = demodata(target_step='detect', target_frame_num=7)
            >>> detections, masks = self.apply(img)
            >>> print('detections = {!r}'.format(detections))
            >>> draw_img = DrawHelper.draw_detections(img, detections, masks)
            >>> from matplotlib import pyplot as plt
            >>> plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            >>> plt.gca().grid(False)
            >>> plt.show()
        """
        masks = {}
        # Downsample and convert to grayscale
        img_, upfactor = self.preprocess_image(img)

        # Run detection / update background model
        mask = self.background_model.apply(img_)
        if return_info:
            masks['orig'] = mask.copy()

        if self.n_iters < 6:
            # Skip the first few frames while the model is learning
            detections = []
        else:
            if self.config['local_threshold']:
                # Refine initial detections
                mask = self.local_threshold_mask(img_, mask)
                if return_info:
                    masks['local'] = mask.copy()

            # Remove noise
            if self.config['smooth_ksize'] is not None:
                mask = self.postprocess_mask(mask)
                if return_info:
                    masks['post'] = mask.copy()

            # Find detections using CC algorithm
            detections = list(self.masked_detect(mask))

            # Grabcut didn't work that well
            # if False and self.n_iters > 5:
            #     # Refine detections with grabcut
            #     mask = np.zeros(mask.shape, dtype=mask.dtype)
            #     for detection in detections:
            #         print('RUNNING GC')
            #         cc = detection['cc'].astype(np.uint8) * 255
            #         cc = grabcut(img_, cc)
            #         mask[cc > 0] = 255
            #     detections = list(self.masked_detect(mask))
            #     if return_info:
            #         masks['cut'] = mask.copy()

            if self.config['factor'] != 1.0:
                # Upscale back to input img coordinates (to agree with camera calib)
                self.upscale_detections(detections, upfactor)

        self.n_iters += 1
        return detections, masks

    @profile
    def postprocess_mask(self, mask):
        ksize = np.array(self.config['smooth_ksize'])
        ksize = tuple(np.round(ksize / self.config['factor']).astype(np.int))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        # opening is erosion followed by dilation
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, dst=mask)
        # Do a second dilation
        mask = cv2.dilate(src=mask, kernel=kernel, dst=mask)

        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)
        # mask = cv2.erode(src=mask, kernel=kernel, dst=mask)
        # mask = cv2.dilate(src=mask, kernel=kernel, dst=mask)
        return mask

    @profile
    def upscale_detections(self, detections, upfactor):
        """
        inplace upscaling of bounding boxes and points
        (masks are not upscaled)
        """
        for detection in detections:
            center = upfactor * detection['oriented_bbox'].center
            extent = upfactor * detection['oriented_bbox'].extent
            angle = detection['oriented_bbox'].angle
            detection['oriented_bbox'] = OrientedBBox(
                tuple(center), tuple(extent), angle)
            detection['hull'] = upfactor * detection['hull']
            detection['box_points'] = upfactor * detection['box_points']

    @profile
    def preprocess_image(self, img):
        """
        Preprocess image before subtracting backround
        """
        # Convert to grayscale
        img_ = ensure_grayscale(img)
        # Downsample image before running detection
        factor = self.config['factor']
        if factor != 1.0:
            if self.config['bg_algo'] == 'median':
                # Special downsampling for median background subtraction algo
                factor = int(factor)
                img_ = downsample_average_blocks(img_, factor)
                upfactor = np.array([factor, factor])
            else:
                downfactor_ = 1 / factor
                img_, downfactor = imscale(img, downfactor_)
                upfactor = 1 / np.array(downfactor)
        else:
            upfactor = np.array([1., 1.])
            img_ = img
        return img_, upfactor

    @profile
    def masked_detect(self, mask, **kwargs):
        """
        Find pixel locs of each cc and determine if its a valid detection

        Args:
            mask (ndarray): mask where non-zero pixels indicate all candidate
                objects
        """
        img_h, img_w = mask.shape

        # 4-way connected compoment algorithm
        n_ccs, cc_mask = cv2.connectedComponents(mask, connectivity=8)

        hist, bins = np.histogram(cc_mask[cc_mask > 0].ravel(),
                                  bins=np.arange(1, n_ccs + 1))
        # Process only labels with enough points
        filter_size = kwargs.get('filter_size', True)
        if filter_size:
            # speed optimization: quickly determine num pixels for each cc
            # using a histogram instead of checking in the filter func
            factor = self.config['factor']
            min_size = self.config['min_size'] / (factor ** 2)
            # only consider large enough regions
            valid_labels = bins[0:-1][hist >= min_size]
            # no longer need to do it in the child func
            kwargs['filter_size'] = False
        else:
            valid_labels = np.arange(1, n_ccs)

        # Filter ccs to generate only "good" detections
        for cc_label in valid_labels:
            cc = (cc_mask == cc_label)
            detection = self.filter_detection(cc, **kwargs)
            if detection is not None:
                yield detection

    @profile
    def filter_detection(self, cc, filter_size=True, filter_border=True,
                         filter_aspect=True):
        """
        Creates a detection from a CC-mask, or returns None if it is filtered.

        Args:
            cc (ndarray): mask where non-zero pixels indicate a single
                candidate object

        Returns:
            detection: dict or None
                a dictionary containing mask and bounding box information about
                a single object detection. If the conditions are not met,
                returns None.
        """
        # note, `np.where` returns coords in (r, c)

        if filter_size:
            # Remove small regions
            n_pixels = cc.sum()
            factor = self.config['factor']
            min_size = self.config['min_size'] / (factor ** 2)
            if n_pixels < min_size:
                return None

        cc_y, cc_x = np.where(cc)

        if filter_border:
            # Define thresholds to filter edges
            factor = self.config['factor']
            minx_lim, miny_lim = self.config['edge_trim']
            maxx_lim = cc.shape[1] - (minx_lim / factor)
            maxy_lim = cc.shape[0] - (miny_lim / factor)

            # Filter objects detected on the edge of the image region
            minx, maxx = cc_x.min(), cc_x.max()
            miny, maxy = cc_y.min(), cc_y.max()
            if any([minx < minx_lim, maxx > maxx_lim,
                    miny < miny_lim, maxy > maxy_lim]):
                return None

        # generate the valid detection
        points = np.vstack([cc_x, cc_y]).T

        # Find a minimum oriented bounding box around the points
        hull = cv2.convexHull(points)
        oriented_bbox = OrientedBBox(*cv2.minAreaRect(hull))
        w, h = oriented_bbox.extent

        if w == 0 or h == 0:
            return None

        # Filter objects without fishy aspect ratios
        if filter_aspect:
            ar = max([(w / h), (h / w)])
            min_aspect, max_aspect = self.config['aspect_thresh']
            if any([ar < min_aspect, ar > max_aspect]):
                return None

        detection = {
            # 'points': points,
            'box_points': cv2.boxPoints(oriented_bbox),
            'oriented_bbox': oriented_bbox,
            'cc': cc,
            'hull': hull[:, 0, :],
        }
        return detection

    @profile
    def local_threshold_mask(self, img_, post_mask):
        refined_mask = post_mask.copy()
        # refined_mask = np.zeros(post_mask.shape, dtype=np.uint8)

        # Generate a set of inital detections
        detections = list(self.masked_detect(post_mask, filter_border=False,
                                             filter_aspect=False))
        # Sort detections such that the largest detections are processed first,
        # so that the large fish do not remove smaller fish.
        areas = np.array([np.prod(det['oriented_bbox'].extent)
                          for det in detections])
        sortx = np.argsort(areas)
        for detection in np.take(detections, sortx):
            refined_mask = self.refine_local_threshold(img_, refined_mask,
                                                       detection)
        return refined_mask

    @profile
    def refine_local_threshold(self, img_, refined_mask, detection):
        """
        Function to perform local threshold background subtraction on all
        individual fish, with boxed coordinates set by coords.

        Each object, specified is locally background subtracted using a
        threshold equal to the mean + 1*sigma of the gaussian fit to the
        histogram of the grayscale values in tempim.

        imN replaces all the objects in im with the new 0/1 local objects.  imN
        should be a more correct estimate of the actual object sizes.

        Ignore:
            pt.imshow(DrawHelper.draw_detections(img_.astype(np.uint8), [detection], {}))

        Doctest:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> self, img = demodata(target_step='detect', target_frame_num=7)
            >>> img_ = self.preprocess_image(img)[0]
            >>> refined_mask = self.background_model.apply(img_).copy()
            >>> detection = list(self.masked_detect(refined_mask))[0]
        """
        center_c, center_r = map(int, map(round, detection['oriented_bbox'].center))
        # Extract a padded region around the detection
        # Use 2 * the height of the rectangle as the radius
        radius = max(2, int(round(detection['oriented_bbox'].extent[1] * 2)))
        # radius = int(round(max(detection['oriented_bbox'].extent) * 2))
        r1 = max(center_r - radius, 0)
        c1 = max(center_c - radius, 0)
        r2 = min(center_r + radius, img_.shape[0])
        c2 = min(center_c + radius, img_.shape[1])
        chip = img_[r1:r2, c1:c2]

        import scipy.stats
        mu, sigma = scipy.stats.norm.fit(chip.ravel())
        level = mu + sigma

        # Remove objects with a total number of pixels below the set minn at
        # the beginning, while keeping those that are bordering (includes
        # diagonals) the 'fish', which is the largest.
        sub_mask = np.zeros(chip.shape, dtype=np.uint8)
        sub_mask[chip > level] = 255
        sub_ccs, sub_labels = cv2.connectedComponents(sub_mask, connectivity=4)
        hist, bins = np.histogram(sub_labels[sub_labels > 0].ravel(),
                                  bins=np.arange(1, sub_ccs + 1))
        if len(hist) > 0:
            largest_label = bins[hist.argmax()]
        else:
            largest_label = 1

        # Choose only one of these CCs
        refined_mask[r1:r2, c1:c2][sub_labels == largest_label] = 255
        return refined_mask


class MedianBackgroundSubtractor(object):
    """
    algorithm for subtracting background net in fish stereo images
    """

    def __init__(bgmodel, diff_thresh=19):
        bgmodel.diff_thresh = diff_thresh
        bgmodel.bgimg = None

    @profile
    def apply(bgmodel, img):
        """
        Debugging:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> #
            >>> from matplotlib import pyplot as plt
            >>> image_path_list1, _, _ = demodata_input(dataset=2)
            >>> stream = FrameStream(image_path_list1, stride=1)
            >>> detector = FishDetector(bg_algo='median')
            >>> bgmodel = MedianBackgroundSubtractor()
            >>> def getimg(i):
            >>>     return detector.preprocess_image(stream[i][1])[0]
            >>> start = 0
            >>> num = 9
            >>> step = 3
            >>> for i in range(start, start + num, step):
            >>>     imgs = [getimg(i + j) for j in range(step)]
            >>>     masks = []
            >>>     bgs = []
            >>>     for img in imgs:
            >>>         masks.append(bgmodel.apply(img))
            >>>         bgs.append(bgmodel.bgimg.copy())
            >>>     fig = plt.figure(i)
            >>>     for j, (img, mask, bg) in enumerate(zip(imgs, masks, bgs)):
            >>>         ax = fig.add_subplot(step, 3, 1 + step * j)
            >>>         ax.imshow(img, interpolation='nearest', cmap='gray')
            >>>         ax.grid(False)
            >>>         ax = fig.add_subplot(step, 3, 2 + step * j)
            >>>         ax.imshow(mask)
            >>>         ax.grid(False)
            >>>         ax = fig.add_subplot(step, 3, 3 + step * j)
            >>>         ax.imshow(bg, interpolation='nearest', cmap='gray')
            >>>         ax.grid(False)
            >>> plt.show()
        """
        # Subtract the previous background image and make a new one
        if bgmodel.bgimg is None:
            bgmodel.bgimg = img.copy()
            mask = np.zeros(img.shape, dtype=np.uint8)
        else:
            fr_diff = img - bgmodel.bgimg
            mask = fr_diff > bgmodel.diff_thresh

            # This seems to put black pixels always in the background.
            fg_mask = (fr_diff > bgmodel.diff_thresh)
            fg_img = (fg_mask * img)  # this is background substracted image
            mask = (fg_img > 0).astype(np.uint8) * 255

            # median update the background image
            bgmodel.bgimg -= 1
            bgmodel.bgimg[fr_diff > 1] += 2
        return mask


class StereoCalibration(object):
    """
    Helper class for reading / accessing stereo camera calibration params
    """
    def __init__(cal):
        cal.data = None
        cal.unit = 'milimeters'

    def __str__(cal):
        return '{}({})'.format(cal.__class__.__name__, cal.data)

    def extrinsic_vecs(cal):
        rvec1 = cal.data['left']['extrinsic']['om']
        tvec1 = cal.data['right']['extrinsic']['om']

        rvec2 = cal.data['right']['extrinsic']['om']
        tvec2 = cal.data['right']['extrinsic']['T']
        return rvec1, tvec1, rvec2, tvec2

    def distortions(cal):
        kc1 = cal.data['right']['intrinsic']['kc']
        kc2 = cal.data['left']['intrinsic']['kc']
        return kc1, kc2

    def intrinsic_matrices(cal):
        K1 = cal._make_intrinsic_matrix(cal.data['left']['intrinsic'])
        K2 = cal._make_intrinsic_matrix(cal.data['right']['intrinsic'])
        return K1, K2

    def _make_intrinsic_matrix(cal, intrin):
        fc = intrin['fc']
        cc = intrin['cc']
        alpha_c = intrin['alpha_c']
        KK = np.array([
            [fc[0], alpha_c * fc[0], cc[0]],
            [    0,           fc[1], cc[1]],
            [    0,               0,     1],
        ])
        return KK

    @classmethod
    def from_matfile(StereoCalibration, cal_fpath):
        """
        Loads a matlab camera calibration file from disk

        References:
            http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
            http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html

        Doctest:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> _, _, cal_fpath = demodata_input(dataset=1)
            >>> cal = StereoCalibration.from_matfile(cal_fpath)
            >>> print('cal = {}'.format(cal))
            >>> _, _, cal_fpath = demodata_input(dataset=2)
            >>> cal = StereoCalibration.from_matfile(cal_fpath)
            >>> print('cal = {}'.format(cal))
        """
        import scipy.io
        cal_data = scipy.io.loadmat(cal_fpath)
        keys = ['om', 'T', 'fc_left', 'fc_right', 'cc_left', 'cc_right',
                'kc_left', 'kc_right', 'alpha_c_left', 'alpha_c_right']

        if isinstance(cal_data, dict) and 'Cal' in cal_data:
            vals = cal_data['Cal'][0][0]
            flat_dict = {k: v.ravel() for k, v in zip(keys, vals)}
        else:
            flat_dict = {key: cal_data[key].ravel() for key in keys}

        data = {
            'left': {
                'extrinsic': {
                    # Center wold on the left camera
                    'om': np.zeros(3),  # rotation vector
                    'T': np.zeros(3),  # translation vector
                },
                'intrinsic': {
                    'fc': flat_dict['fc_left'],  # focal point
                    'cc': flat_dict['cc_left'],  # principle point
                    'alpha_c': flat_dict['alpha_c_left'][0],  # skew
                    'kc': flat_dict['kc_left'],  # distortion
                }
            },

            'right': {
                'extrinsic': {
                    'om': flat_dict['om'],  # rotation vector
                    'T': flat_dict['T'],  # translation vector
                },
                'intrinsic': {
                    'fc': flat_dict['fc_right'],  # focal point
                    'cc': flat_dict['cc_right'],  # principle point
                    'alpha_c': flat_dict['alpha_c_right'][0],  # skew
                    'kc': flat_dict['kc_right'],  # distortion
                }
            },
        }
        cal = StereoCalibration()
        cal.data = data
        return cal


class FishStereoTriangulationAssignment(object):
    def __init__(self, **kwargs):
        self.config = {
            # Threshold for errors between before & after projected
            # points to make matches between left and right
            'max_err': [6, 14],
            # 'max_err': [300, 300],
            'small_len': 15,  # in centimeters
        }
        self.config.update(kwargs)

    @profile
    def triangulate(self, cal, det1, det2):
        """
        Assuming, det1 matches det2, we determine 3d-coordinates of each
        detection and measure the reprojection error.

        References:
            http://answers.opencv.org/question/117141
            https://gist.github.com/royshil/7087bc2560c581d443bc
            https://stackoverflow.com/a/29820184/887074

        Doctest:
            >>> # Rows are detections in img1, cols are detections in img2
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> detections1, detections2, cal = demodata(target_step='triangulate', target_frame_num=6)
            >>> det1, det2 = detections1[0], detections2[0]
            >>> self = FishStereoTriangulationAssignment()
            >>> assignment, assign_data, cand_errors = self.triangulate(cal, det1, det2)
        """
        _debug = 0
        if _debug:
            # Use 4 corners and center to ensure matrix math is good
            # (hard to debug when ndims == npts, so make npts >> ndims)
            pts1 = np.vstack([det1['box_points'], det1['oriented_bbox'].center])
            pts2 = np.vstack([det2['box_points'], det2['oriented_bbox'].center])
        else:
            # Use only the corners of the bbox
            pts1 = det1['box_points'][[0, 2]]
            pts2 = det2['box_points'][[0, 2]]

        # Move into opencv point format (num x 1 x dim)
        pts1_cv = pts1[:, None, :]
        pts2_cv = pts2[:, None, :]

        # Grab camera parameters
        K1, K2 = cal.intrinsic_matrices()
        kc1, kc2 = cal.distortions()
        rvec1, tvec1, rvec2, tvec2 = cal.extrinsic_vecs()

        # Make extrincic matrices
        R1 = cv2.Rodrigues(rvec1)[0]
        R2 = cv2.Rodrigues(rvec2)[0]
        T1 = tvec1[:, None]
        T2 = tvec2[:, None]
        RT1 = np.hstack([R1, T1])
        RT2 = np.hstack([R2, T2])

        # Undistort points
        # This puts points in "normalized camera coordinates" making them
        # independent of the intrinsic parameters. Moving to world coordinates
        # can now be done using only the RT transform.
        unpts1_cv = cv2.undistortPoints(pts1_cv, K1, kc1)
        unpts2_cv = cv2.undistortPoints(pts2_cv, K2, kc2)

        # unpts1_cv = self.normalizePixel(pts1.T, **cal.data['left']['intrinsic'])
        # unpts2_cv = self.normalizePixel(pts2.T, **cal.data['right']['intrinsic'])

        # note: trinagulatePoints docs say that it wants a 3x4 projection
        # matrix (ie K.dot(RT)), but we only need to use the RT extrinsic
        # matrix because the undistorted points already account for the K
        # intrinsic matrix.
        world_pts_homog = cv2.triangulatePoints(RT1, RT2, unpts1_cv, unpts2_cv)
        world_pts = from_homog(world_pts_homog)

        # Compute distance between 3D bounding box points
        if _debug:
            corner1, corner2 = world_pts.T[[0, 2]]
        else:
            corner1, corner2 = world_pts.T

        # Convert to centimeters
        fishlen = np.linalg.norm(corner1 - corner2) / 10

        # Reproject points
        world_pts_cv = world_pts.T[:, None, :]
        proj_pts1_cv = cv2.projectPoints(world_pts_cv, rvec1, tvec1, K1, kc1)[0]
        proj_pts2_cv = cv2.projectPoints(world_pts_cv, rvec2, tvec2, K2, kc2)[0]

        # Check error
        err1 = ((proj_pts1_cv - pts1_cv)[:, 0, :] ** 2).sum(axis=1)
        err2 = ((proj_pts2_cv - pts2_cv)[:, 0, :] ** 2).sum(axis=1)
        errors = np.hstack([err1, err2])

        # Get 3d points in each camera's reference frame
        # Note RT1 is the identity and RT are 3x4, so no need for `from_homog`
        # Return points in with shape (N,3)
        pts1_3d = RT1.dot(to_homog(world_pts)).T
        pts2_3d = RT2.dot(to_homog(world_pts)).T
        return pts1_3d, pts2_3d, errors, fishlen

    def normalizePixel(self, pts, fc, cc, kc, alpha_c):
        """
        Alternative to cv2.undistortPoints. The main difference is that this
        runs iterative distortion componsation for 20 iters instead of 5.

        Ultimately, it doesn't make much difference, use opencv instead because
        its faster.
        """
        x_distort = np.array([(pts[0, :] - cc[0]) / fc[0], (pts[1, :] - cc[1]) / fc[1]])
        x_distort[0, :] = x_distort[0, :] - alpha_c * x_distort[1, :]
        if not np.linalg.norm(kc) == 0:
            xn = self.compDistortion(x_distort, kc)
        else:
            xn = x_distort
        return xn

    def compDistortion(self, xd, k):
        if len(k) == 1:  # original comp_distortion_oulu
            r_2 = xd[:, 0]**2 + xd[:, 1]**2
            radial_d = 1 + np.dot(np.ones((2, 1)), np.array([(k * r_2)]))
            radius_2_comp = r_2 / radial_d[0, :]
            radial_d = 1 + np.dot(np.ones((2, 1)), np.array([(k * radius_2_comp)]))
            # x = x_dist / radial_d

        else:  # original comp_distortion_oulu
            k1 = k[0]
            k2 = k[1]
            k3 = k[4]
            p1 = k[2]
            p2 = k[3]

            x = xd

            for kk in range(20):
                d = x**2
                r_2 = d.sum(axis=0)
                k_radial = 1 + k1 * r_2 + k2 * r_2**2 + k3 * r_2**3
                delta_x = np.array([2 * p1 * x[0, :] * x[1, :] + p2 * (r_2 + 2 * x[0, :]**2),
                                    p1 * (r_2 + 2 * x[0, :]**2) + 2 * p2 * x[0, :] * x[1, :]])
                x = (xd - delta_x) / (np.dot(np.ones((2, 1)), np.array([k_radial])))
            return x

    def minimum_weight_assignment(self, cost_errors):
        """
        Finds optimal assignment of left-camera to right-camera detections

        Doctest:
            >>> # Rows are detections in img1, cols are detections in img2
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> self = FishStereoTriangulationAssignment()
            >>> cost_errors = np.array([
            >>>     [9, 2, 1, 9],
            >>>     [4, 1, 5, 5],
            >>>     [9, 9, 2, 4],
            >>> ])
            >>> assign1 = self.minimum_weight_assignment(cost_errors)
            >>> assign2 = self.minimum_weight_assignment(cost_errors.T)
        """
        n1, n2 = cost_errors.shape
        n = max(n1, n2)
        # Embed the [n1 x n2] matrix in a padded (with inf) [n x n] matrix
        cost_matrix = np.full((n, n), fill_value=np.inf)
        cost_matrix[0:n1, 0:n2] = cost_errors

        # Find an effective infinite value for infeasible assignments
        is_infeasible = np.isinf(cost_matrix)
        is_positive = cost_matrix > 0
        feasible_vals = cost_matrix[~(is_infeasible & is_positive)]
        large_val = (n + feasible_vals.sum()) * 2
        # replace infinite values with effective infinite values
        cost_matrix[is_infeasible] = large_val

        # Solve munkres problem for minimum weight assignment
        indexes = list(zip(*scipy.optimize.linear_sum_assignment(cost_matrix)))
        # Return only the feasible assignments
        assignment = [(i, j) for (i, j) in indexes
                      if cost_matrix[i, j] < large_val]
        return assignment

    @profile
    def find_matches(self, cal, detections1, detections2):
        """
        Match detections from the left camera to detections in the right camera

        Doctest:
            >>> # Rows are detections in img1, cols are detections in img2
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from gmm_online_background_subtraction import *
            >>> detections1, detections2, cal = demodata(target_step='triangulate', target_frame_num=6)
            >>> self = FishStereoTriangulationAssignment()
            >>> assignment, assign_data, cand_errors = self.find_matches(cal, detections1, detections2)
        """
        n_detect1, n_detect2 = len(detections1), len(detections2)
        cand_world_pts = {}
        cand_fishlen = {}
        cand_rang = {}

        # Initialize matrix of reprojection errors
        cost_errors = np.full((n_detect1, n_detect2), fill_value=np.inf)
        cand_errors = np.full((n_detect1, n_detect2), fill_value=np.inf)

        # Find the liklihood that each pair of detections matches by
        # triangulating and then measuring the reprojection error.
        for (i, det1), (j, det2) in it.product(enumerate(detections1),
                                               enumerate(detections2)):
            # Triangulate assuming det1 and det2 match, but return the
            # reprojection error so we can check if this assumption holds
            pts1_3d, pts2_3d, errors, fishlen = self.triangulate(cal, det1, det2)
            error = errors.mean()

            # Mark the pair (i, j) as a potential candidate match
            cand_world_pts[(i, j)] = pts1_3d
            cand_errors[i, j] = error
            cand_fishlen[(i, j)] = fishlen
            cand_rang[(i, j)] = pts1_3d.T[2].mean()

            # Check chirality
            # Both Z-coordinates must be positive (i.e. in front the cameras)
            z_coords1 = pts1_3d.T[2]
            z_coords2 = pts2_3d.T[2]
            both_in_front = np.all(z_coords1 > 0) and np.all(z_coords2 > 0)
            if not both_in_front:
                # Ignore out-of-view correspondences
                continue

            # Check if reprojection error is too high
            max_error = self.config['max_err']
            small_len = self.config['small_len']  # hardcoded to 15cm in matlab version
            if len(max_error) == 2:
                error_thresh = max_error[0] if fishlen <= small_len else max_error[1]
            else:
                error_thresh = max_error[0]

            if error  >= error_thresh:
                # Ignore correspondences with high reprojection error
                continue

            cost_errors[i, j] = error

        # Find the matching with minimum reprojection error, such that each
        # detection in one camera can match at most one detection in the other.
        assignment = self.minimum_weight_assignment(cost_errors)

        # get associated data with each assignment
        assign_data = [
            {
                'ij': (i, j),
                'fishlen': cand_fishlen[(i, j)],
                'error': cand_errors[(i, j)],
                'range': cand_rang[(i, j)],
            }
            for i, j in assignment
        ]
        return assignment, assign_data, cand_errors


class DrawHelper(object):
    """
    Visualization of the algorithm stages
    """

    @staticmethod
    @profile
    def draw_detections(img, detections, masks):
        # Upscale masks to original image size
        dsize = tuple(img.shape[0:2][::-1])
        shape = img.shape[0:2]

        # Align all masks with the image size
        masks2 = {
            k: cv2.resize(v, dsize, interpolation=cv2.INTER_NEAREST)
            for k, v in masks.items()
        }

        # Create a heatmap for detections
        draw_mask = np.zeros(shape, dtype=np.float)

        if 'orig' in masks2:
            draw_mask[masks2['orig'] > 0] = .4
        if 'local' in masks2:
            draw_mask[masks2['local'] > 0] = .65
        if 'post' in masks2:
            draw_mask[masks2['post'] > 0] = .85

        for n, detection in enumerate(detections, start=1):
            cc = cv2.resize(detection['cc'].astype(np.uint8), dsize)
            draw_mask[cc > 0] = 1.0
        draw_img = overlay_heatmask(img, draw_mask, alpha=.7)

        # Draw bounding boxes and contours
        for detection in detections:
            # Points come back in (x, y), but we want to draw in (r, c)
            box_points = np.round(detection['box_points']).astype(np.int)
            hull_points = np.round(detection['hull']).astype(np.int)
            draw_img = cv2.drawContours(
                image=draw_img, contours=[hull_points], contourIdx=-1,
                color=(255, 0, 0), thickness=2)
            draw_img = cv2.drawContours(
                image=draw_img, contours=[box_points], contourIdx=-1,
                color=(0, 255, 0), thickness=2)
        return draw_img

    @staticmethod
    @profile
    def draw_stereo_detections(img1, detections1, masks1,
                               img2, detections2, masks2,
                               assignment=None, assign_data=None,
                               cand_errors=None):
        import textwrap
        BGR_RED = (0, 0, 255)
        line_color = BGR_RED
        text_color = BGR_RED

        draw1 = DrawHelper.draw_detections(img1, detections1, masks1)
        draw2 = DrawHelper.draw_detections(img2, detections2, masks2)
        stacked = np.hstack([draw1, draw2])

        def putMultiLineText(img, text, org, **kwargs):
            """
            References:
                https://stackoverflow.com/questions/27647424/
            """
            getsize_kw = {
                k: kwargs[k]
                for k in ['fontFace', 'fontScale', 'thickness']
                if k in kwargs
            }
            x0, y0 = org
            ypad = kwargs.get('thickness', 2) + 4
            y = y0
            for i, line in enumerate(text.split('\n')):
                (w, h), text_sz = cv2.getTextSize(text, **getsize_kw)
                img = cv2.putText(img, line, (x0, y), **kwargs)
                y += (h + ypad)
            return img

        if assignment is not None and len(cand_errors) > 0:
            for j in range(cand_errors.shape[1]):
                i = np.argmin(cand_errors[:, j])
                if (i, j) in assignment or (j, i) in assignment:
                    continue
                error = cand_errors[i, j]
                # if not np.isinf(error):
                center1 = np.array(detections1[i]['oriented_bbox'].center)
                center2 = np.array(detections2[j]['oriented_bbox'].center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int))
                center2_ = tuple(center2_.astype(np.int))

                BGR_PURPLE = (255, 0, 255)

                stacked = cv2.line(stacked, center1, center2_,
                                   color=BGR_PURPLE,
                                   lineType=cv2.LINE_AA,
                                   thickness=1)
                text = textwrap.dedent(
                    '''
                    error = {error:.2f}
                    '''
                ).strip().format(error=error)

                stacked = putMultiLineText(stacked, text, org=center2_,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=BGR_PURPLE,
                                           thickness=2, lineType=cv2.LINE_AA)

            for (i, j), info in zip(assignment, assign_data):

                center1 = np.array(detections1[i]['oriented_bbox'].center)
                center2 = np.array(detections2[j]['oriented_bbox'].center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int))
                center2_ = tuple(center2_.astype(np.int))

                stacked = cv2.line(stacked, center1, center2_, color=line_color,
                                   lineType=cv2.LINE_AA,
                                   thickness=2)

                text = textwrap.dedent(
                    '''
                    len = {fishlen:.2f}cm
                    error = {error:.2f}
                    range = {range:.2f}mm
                    '''
                ).strip().format(**info)

                stacked = putMultiLineText(stacked, text, org=center1,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=text_color,
                                           thickness=2, lineType=cv2.LINE_AA)

        with_orig = False
        if with_orig:
            # Put in the original images
            bottom = np.hstack([img1, img2])
            stacked = np.vstack([stacked, bottom])

        return stacked


# -----------------
# Testing functions
# -----------------


class FrameStream(object):
    """
    Helper for iterating through a sequence of image frames
    """
    def __init__(stream, image_path_list, stride=1):
        stream.image_path_list = image_path_list
        stream.stride = stride
        stream.length = len(image_path_list)

    def __len__(stream):
        return stream.length

    def __getitem__(stream, index):
        img_fpath = stream.image_path_list[index]
        frame_id = basename(img_fpath).split('_')[0]
        img = cv2.imread(img_fpath)
        return frame_id, img

    def __iter__(stream):
        for i in range(0, len(stream), stream.stride):
            yield stream[i]


def demodata_input(dataset=1):
    import glob

    if dataset == 1:
        data_fpath = expanduser('~/data/autoprocess_test_set')
        cal_fpath = join(data_fpath, 'cal_201608.mat')
        img_path1 = join(data_fpath, 'image_data/left')
        img_path2 = join(data_fpath, 'image_data/right')
    elif dataset == 2:
        data_fpath = expanduser('~/data/camtrawl_stereo_sample_data/')
        # cal_fpath = join(data_fpath, 'code/Calib_Results_stereo_1608.mat')
        cal_fpath = join(data_fpath, 'code/cal_201608.mat')
        img_path1 = join(data_fpath, 'Haul_83/left')
        img_path2 = join(data_fpath, 'Haul_83/right')
    else:
        assert False, 'bad dataset'

    image_path_list1 = sorted(glob.glob(join(img_path1, '*.jpg')))
    image_path_list2 = sorted(glob.glob(join(img_path2, '*.jpg')))
    assert len(image_path_list1) == len(image_path_list2)
    return image_path_list1, image_path_list2, cal_fpath


def demodata(dataset=1, target_step='detect', target_frame_num=7):
    """
    Helper for doctests. Gets test data at different points in the pipeline.
    """
    if 'target_step' not in vars():
        target_step = 'detect'
    if 'target_frame_num' not in vars():
        target_frame_num = 7
    image_path_list1, image_path_list2, cal_fpath = demodata_input(dataset=dataset)

    cal = StereoCalibration.from_matfile(cal_fpath)

    detector1 = FishDetector()
    detector2 = FishDetector()
    for frame_num, (img_fpath1, img_fpath2) in enumerate(zip(image_path_list1,
                                                             image_path_list2)):

        frame_id1 = basename(img_fpath1).split('_')[0]
        frame_id2 = basename(img_fpath2).split('_')[0]
        assert frame_id1 == frame_id2
        frame_id = frame_id1
        img1 = cv2.imread(img_fpath1)
        img2 = cv2.imread(img_fpath2)

        if frame_num == target_frame_num:
            if target_step == 'detect':
                return detector1, img1

        detections1, masks1 = detector1.apply(img1)
        detections2, masks2 = detector2.apply(img2)

        n_detect1, n_detect2 = len(detections1), len(detections2)
        print('frame_num, (n_detect1, n_detect2) = {} ({}, {})'.format(
            frame_num, n_detect1, n_detect2))

        if frame_num == target_frame_num:
            # import vtool as vt
            import ubelt as ub
            # stacked = vt.stack_images(masks1['draw'], masks2['draw'], vert=False)[0]
            # stacked = np.hstack([masks1['draw'], masks2['draw']])
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2)
            dpath = ub.ensuredir('out')
            cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_num), stacked)
            cv2.imwrite(dpath + '/mask{}_{}_draw.png'.format(frame_id, frame_num), stacked)
            # return detections1, detections2
            break

    return detections1, detections2, cal


def demo():
    import ubelt as ub
    dataset = 1
    dataset = 2

    image_path_list1, image_path_list2, cal_fpath = demodata_input(dataset=dataset)
    cal = StereoCalibration.from_matfile(cal_fpath)

    dpath = ub.ensuredir('out_{}'.format(dataset))

    bg_algo = 'median'
    bg_algo = 'gmm'

    if bg_algo == 'gmm':
        # Use GMM based model
        stride = 1
        gmm_params = {
            'bg_algo': bg_algo,
            'n_training_frames': 9999,
            # 'gmm_thresh': 20,
            'gmm_thresh': 30,
            'factor': 4,
            'min_size': 800,
            'edge_trim': [10, 10],
            # 'smooth_ksize': None,
            # 'smooth_ksize': (3, 3),
            'smooth_ksize': (10, 10),  # wrt original image size
            'local_threshold': False,
        }
        triangulate_params = {
            'max_err': [200, 200],
        }
        detector1 = FishDetector(**gmm_params)
        detector2 = FishDetector(**gmm_params)
    elif bg_algo == 'median':
        # Use median update difference based model
        detect_params = {
            'bg_algo': bg_algo,
            'factor': 4,
            'min_size': 1500,
            'edge_trim': [10, 10],
            'smooth_ksize': (10, 10),
            'local_threshold': True,
        }
        triangulate_params = {
            # 'max_err': [6, 14],
            'max_err': [200, 200],
        }
        stride = 2
        detector1 = FishDetector(diff_thresh=19, **detect_params)
        detector2 = FishDetector(diff_thresh=15, **detect_params)

    triangulator = FishStereoTriangulationAssignment(**triangulate_params)

    import pprint
    print('Detector1 Config: ' + pprint.pformat(detector1.config, indent=4))
    print('Detector2 Config: ' + pprint.pformat(detector2.config, indent=4))
    print('Triangulate Config: ' + pprint.pformat(triangulator.config, indent=4))

    pprint.pformat(detector2.config)

    stream1 = FrameStream(image_path_list1, stride=stride)
    stream2 = FrameStream(image_path_list2, stride=stride)

    n_total = 0
    all_errors = []
    all_lengths = []

    import ubelt as ub
    prog = ub.ProgIter(enumerate(zip(stream1, stream2)),
                       clearline=True,
                       length=len(stream1) // stride,
                       adjust=False)
    _iter = prog
    # _iter = enumerate(zip(stream1, stream2))
    for frame_num, ((frame_id1, img1), (frame_id2, img2)) in _iter:
        assert frame_id1 == frame_id2
        frame_id = frame_id1
        # prog.ensure_newline()
        # print('frame_id = {!r}'.format(frame_id))

        detections1, masks1 = detector1.apply(img1)
        detections2, masks2 = detector2.apply(img2)

        # stacked = vt.stack_images(masks1['draw'], masks2['draw'], vert=False)[0]
        if len(detections1) > 0 or len(detections2) > 0:
            assignment, assign_data, cand_errors = triangulator.find_matches(
                cal, detections1, detections2)
            all_errors += [d['error'] for d in assign_data]
            all_lengths += [d['fishlen'] for d in assign_data]
            n_total += len(assignment)
            # if len(assignment):
            #     prog.ensure_newline()
            #     print('n_total = {!r}'.format(n_total))
        else:
            cand_errors = None
            assignment, assign_data = None, None

        # if assignment:
        DRAWING = 1
        if DRAWING:
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2,
                                                        assignment, assign_data,
                                                        cand_errors)
            cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_id), stacked)
        # if frame_num == 7:
        #     break
        # if frame_num > 10:
        #     break

    all_errors = np.array(all_errors)
    all_lengths = np.array(all_lengths)
    print('n_total = {!r}'.format(n_total))
    print('ave_error = {:.2f} +- {:.2f}'.format(all_errors.mean(), all_errors.std()))
    print('ave_lengths = {:.2f} +- {:.2f} '.format(all_lengths.mean(), all_lengths.std()))


if __name__ == '__main__':
    demo()
    # import utool as ut
    # ut.dump_profile_text()
