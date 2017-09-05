# -*- coding: utf-8 -*-
"""
Reimplementation of matlab algorithms for fishlength detection in python.
The next step is to move them into kwiver.

TODO:
    fix hard coded paths for doctests
"""
from __future__ import division, print_function
from collections import namedtuple
import cv2
import itertools as it
import numpy as np
import scipy.optimize
from imutils import (
    imscale, ensure_grayscale, overlay_heatmask, from_homog, to_homog,
    putMultiLineText)
from os.path import expanduser, basename, join, splitext
import ubelt as ub


OrientedBBox = namedtuple('OrientedBBox', ('center', 'extent', 'angle'))


class BoundingBox(ub.NiceRepr):
    def __init__(bbox, coords):
        bbox.coords = coords

    def __nice__(self):
        return 'center={}, wh={}'.format(self.center, self.size)

    @classmethod
    def from_coords(self, xmin, ymin, xmax, ymax):
        coords = np.array([xmin, ymin, xmax, ymax])
        return BoundingBox(coords)

    @property
    def lower_left(bbox):
        return bbox.coords[0:2]

    @property
    def upper_right(bbox):
        return bbox.coords[2:4]

    @property
    def center(self):
        if self.coords is None:
            return None
        (xmin, ymin, xmax, ymax) = self.coords
        cx = (xmax + xmin) / 2
        cy = (ymax + ymin) / 2
        return cx, cy

    @property
    def size(self):
        if self.coords is None:
            return None
        (xmin, ymin, xmax, ymax) = self.coords
        width = xmax - xmin
        height = ymax - ymin
        return width, height

    def polygon_points(self):
        (xmin, ymin, xmax, ymax) = self.coords
        return np.array([(xmin, ymin), (xmin, ymax),
                         (xmax, ymax), (xmax, ymin)])

    def scale(self, factor):
        """
        inplace upscaling of bounding boxes and points
        (masks are not upscaled)
        """
        self.coords = np.array(self.coords) * factor
        # center = upfactor * detection['oriented_bbox'].center
        # extent = upfactor * detection['oriented_bbox'].extent
        # angle = detection['oriented_bbox'].angle
        # detection['oriented_bbox'] = OrientedBBox(
        #     tuple(center), tuple(extent), angle)
        # detection['hull'] = upfactor * detection['hull']
        # detection['box_points'] = upfactor * detection['box_points']


class DetectedObject(ub.NiceRepr):
    def __init__(self, bbox, mask):
        if isinstance(bbox, BoundingBox):
            self.bbox = bbox
        else:
            self.bbox = BoundingBox(bbox)
        self.mask = mask

    def __nice__(self):
        return self.bbox.__nice__()

    @classmethod
    def from_connected_component(DetectedObject, cc_mask):
        """
        Example:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from camtrawl_algos import *
            >>> rng = np.random.RandomState(0)
            >>> cc_mask = rng.rand(7, 7) < .1
            >>> DetectedObject.from_connected_component(cc_mask)
        """
        ys, xs = np.where(cc_mask)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        bbox = BoundingBox.from_coords(xmin, ymin, xmax, ymax)
        mask = cc_mask[xmin:xmax, ymin:ymax]
        self = DetectedObject(bbox, mask)
        return self


class GMMForegroundObjectDetector(object):
    """
    Uses background subtraction and 4-way connected compoments algorithm to
    detect potential fish objects. Objects are filtered by size, aspect ratio,
    and closeness to the image border to remove bad detections.

    References:
        https://stackoverflow.com/questions/37300698/gaussian-mixture-model
        http://docs.opencv.org/trunk/db/d5c/tutorial_py_bg_subtraction.html
    """
    def __init__(detector, **kwargs):
        detector.config = {
            'factor': 2.0,
            'smooth_ksize': (10, 10),  # wrt original image size
            # number of frames before the background model is ready
            'n_startup_frames': 1,
            'min_num_pixels': 100,  # wrt original image
            'n_training_frames': 30,
            'gmm_thresh': 30,
            # limits accepable targets to be within this region [padx, pady]
            # These are wrt the original image size
            'edge_trim': [12, 12],
            'aspect_thresh': (3.5, 7.5,),
        }
        detector.config.update(kwargs)
        detector.background_model = cv2.createBackgroundSubtractorMOG2(
            history=detector.config['n_training_frames'],
            varThreshold=detector.config['gmm_thresh'],
            detectShadows=False)
        detector.n_iters = 0
        detector._masks = {}  # kept in memory for visualization

    def detect(detector, img):
        """
        Main algorithm step.
        Detects the objects in the image and update the background model.

        Args:
            img (ndarray): image to perform detection on

        Returns:
            detections : list of DetectedObjects

        Doctest:
            >>> % pylab qt5
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from camtrawl_algos import *
            >>> detector, img = demodata_detections(target_step='detect', target_frame_num=7)
            >>> detections = detector.detect(img)
            >>> print('detections = {!r}'.format(detections))
            >>> masks = detector._masks
            >>> draw_img = DrawHelper.draw_detections(img, detections, masks)
            >>> from matplotlib import pyplot as plt
            >>> plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            >>> plt.gca().grid(False)
            >>> plt.show()
        """
        detector._masks = {}
        # Downsample and convert to grayscale
        img_, upfactor = detector.preprocess_image(img)

        # Run detection / update background model
        mask = detector.background_model.apply(img_)
        detector._masks['orig'] = mask.copy()

        if detector.n_iters < detector.config['n_startup_frames']:
            # Skip the first few frames while the model is learning
            detections = []
        else:
            # Remove noise
            if detector.config['smooth_ksize'] is not None:
                mask = detector.postprocess_mask(mask)
                detector._masks['post'] = mask.copy()

            # Find detections using CC algorithm
            detections = list(detector.detections_in_mask(mask))

            # Upscale back to input img coordinates (to agree with camera calib)
            if detector.config['factor'] != 1.0:
                for detection in detections:
                    detection.bbox.scale(upfactor)

        detector.n_iters += 1
        return detections

    def preprocess_image(detector, img):
        """
        Preprocess image before subtracting backround
        """
        # Convert to grayscale
        img_ = ensure_grayscale(img)
        # Downsample image before running detection
        factor = detector.config['factor']
        if factor != 1.0:
            downfactor_ = 1 / factor
            img_, downfactor = imscale(img, downfactor_)
            upfactor = 1 / downfactor[0]
        else:
            upfactor = 1.0
            img_ = img
        return img_, upfactor

    def postprocess_mask(detector, mask):
        """ remove noise from detection intensity masks """
        ksize = np.array(detector.config['smooth_ksize'])
        ksize = tuple(np.round(ksize / detector.config['factor']).astype(np.int))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        # opening is erosion followed by dilation
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, dst=mask)
        # Do a second dilation
        mask = cv2.dilate(src=mask, kernel=kernel, dst=mask)
        return mask

    def detections_in_mask(detector, mask):
        """
        Find pixel locs of each cc and determine if its a valid detection

        Args:
            mask (ndarray): mask where non-zero pixels indicate all candidate
                objects
        """
        # 4-way connected compoment algorithm
        n_ccs, cc_mask = cv2.connectedComponents(mask, connectivity=8)

        # Process only labels with enough points
        min_num_pixels = detector.config['min_num_pixels']
        if min_num_pixels is None:
            valid_labels = np.arange(1, n_ccs)
        else:
            # speed optimization: quickly determine num pixels for each cc
            # using a histogram instead of checking in the filter func
            hist, bins = np.histogram(cc_mask[cc_mask > 0].ravel(),
                                      bins=np.arange(1, n_ccs + 1))
            factor = detector.config['factor']
            min_num_pixels = min_num_pixels / (factor ** 2)
            # only consider large enough regions
            valid_labels = bins[0:-1][hist >= min_num_pixels]

        # Filter ccs to generate only "good" detections
        for cc_label in valid_labels:
            cc = (cc_mask == cc_label)
            detection = DetectedObject.from_connected_component(cc)
            yield detection


class FishDetectionFilter(object):
    def __init__(self, **kwargs):
        self.config = {
            'edge_trim': [12, 12],
            # Min/Max aspect ratio for filtering out non-fish objects
            'aspect_thresh': (3.5, 7.5,),
            'min_n_pixels': 100,
        }

    def filter_detections(self, detections, img_dsize):
        for detection in detections:
            if self.is_valid(detection, img_dsize):
                yield detection

    def is_valid(self, detection, img_dsize):
        """
        Checks if the detection passes filtering constraints

        Args:
            detection (DetectedObject): mask where non-zero pixels indicate a single
                candidate object

        Returns:
            bool: True if the detection is valid else False
        """
        # note, `np.where` returns coords in (r, c)

        if self.config['min_n_pixels'] is not None:
            # Remove small regions
            n_pixels = (detection.mask > 0).sum()
            factor = 2  # HACK
            min_n_pixels = self.config['min_n_pixels'] / (factor ** 2)
            if n_pixels < min_n_pixels:
                return False

        cc_y, cc_x = np.where(detection.mask)

        if self.config['edge_trim'] is not None:
            # Define thresholds to filter edges
            # img_width, img_height = img_dsize
            # factor = self.config['factor']
            factor = 2  # HACK
            xmin_lim, ymin_lim = self.config['edge_trim']
            img_width, img_height = img_dsize
            xmax_lim = img_width - (xmin_lim / factor)
            ymax_lim = img_height - (ymin_lim / factor)

            # Filter objects detected on the edge of the image region
            (xmin, ymin, xmax, ymax) = detection.bbox.coords
            if any([xmin < xmin_lim, xmax > xmax_lim,
                    ymin < ymin_lim, ymax > ymax_lim]):
                return None

        # generate the valid detection
        points = np.vstack([cc_x, cc_y]).T

        # Find a minimum oriented bounding box around the points
        hull = cv2.convexHull(points)
        oriented_bbox = OrientedBBox(*cv2.minAreaRect(hull))
        w, h = oriented_bbox.extent

        if w == 0 or h == 0:
            return False

        # Filter objects without fishy aspect ratios
        ar = max([(w / h), (h / w)])
        min_aspect, max_aspect = self.config['aspect_thresh']
        if any([ar < min_aspect, ar > max_aspect]):
            return False

        return True


class FishStereoTriangulationAssignment(object):
    def __init__(self, **kwargs):
        self.config = {
            # Threshold for errors between before & after projected
            # points to make matches between left and right
            'max_err': [6, 14],
            # 'max_err': [300, 300],
            'small_len': 150,  # in milimeters
        }
        self.config.update(kwargs)

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
            >>> from camtrawl_algos import *
            >>> detections1, detections2, cal = demodata_detections(target_step='triangulate', target_frame_num=6)
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

        # Length is in milimeters
        fishlen = np.linalg.norm(corner1 - corner2)

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

    def minimum_weight_assignment(self, cost_errors):
        """
        Finds optimal assignment of left-camera to right-camera detections

        Doctest:
            >>> # Rows are detections in img1, cols are detections in img2
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from camtrawl_algos import *
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

    def find_matches(self, cal, detections1, detections2):
        """
        Match detections from the left camera to detections in the right camera

        Doctest:
            >>> # Rows are detections in img1, cols are detections in img2
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from camtrawl_algos import *
            >>> detections1, detections2, cal = demodata_detections(target_step='triangulate', target_frame_num=6)
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
            small_len = self.config['small_len']  # hardcoded to 15cm in matlab
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

    @staticmethod
    def _make_intrinsic_matrix(intrin):
        """ convert intrinsic dict to matrix """
        fc = intrin['fc']
        cc = intrin['cc']
        alpha_c = intrin['alpha_c']
        KK = np.array([
            [fc[0], alpha_c * fc[0], cc[0]],
            [    0,           fc[1], cc[1]],
            [    0,               0,     1],
        ])
        return KK

    @staticmethod
    def _make_intrinsic_params(K):
        """ convert intrinsic matrix to dict """
        intrin = {}
        fc = intrin['fc'] = np.zeros(2)
        cc = intrin['cc'] = np.zeros(2)
        [[fc[0], alpha_c_fc0, cc[0]],
         [    _,       fc[1], cc[1]],
         [    _,           _,     _]] = K
        intrin['alpha_c'] = np.array([alpha_c_fc0 / fc[0]])
        return intrin

    @classmethod
    def from_file(StereoCalibration, cal_fpath):
        ext = splitext(cal_fpath)[1].lower()
        if ext == '.mat':
            return StereoCalibration.from_matfile(cal_fpath)
        elif ext == '.npz':
            return StereoCalibration.from_npzfile(cal_fpath)
        else:
            raise ValueError('unknown extension {}'.format(ext))

    @classmethod
    def from_npzfile(StereoCalibration, cal_fpath):
        """
        Doctest:
            >>> import sys
            >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
            >>> from camtrawl_algos import *
            >>> cal_fpath = '/home/joncrall/data/camtrawl_stereo_sample_data/201608_calibration_data/selected/Camtrawl_2016.npz'
            >>> cal = StereoCalibration.from_npzfile(cal_fpath)
        """
        data = dict(np.load(cal_fpath))
        flat_dict = {}
        flat_dict['om'] = cv2.Rodrigues(data['R'])[0].ravel()
        flat_dict['T'] = data['T'].ravel()

        K1 = data['cameraMatrixL']
        intrin1 = StereoCalibration._make_intrinsic_params(K1)
        flat_dict['fc_left'] = intrin1['fc']
        flat_dict['cc_left'] = intrin1['cc']
        flat_dict['alpha_c_left'] = intrin1['alpha_c']
        flat_dict['kc_left'] = data['distCoeffsL'].ravel()

        K2 = data['cameraMatrixR']
        intrin2 = StereoCalibration._make_intrinsic_params(K2)
        flat_dict['fc_right'] = intrin2['fc']
        flat_dict['cc_right'] = intrin2['cc']
        flat_dict['alpha_c_right'] = intrin2['alpha_c']
        flat_dict['kc_right'] = data['distCoeffsR'].ravel()
        return StereoCalibration._from_flat_dict(flat_dict)

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
            >>> from camtrawl_algos import *
            >>> _, _, cal_fpath = demodata_input(dataset='test')
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
        return StereoCalibration._from_flat_dict(flat_dict)

    @classmethod
    def _from_flat_dict(StereoCalibration, flat_dict):
        """ helper used by matlab and numpy readers """
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


class DrawHelper(object):
    """
    Visualization of the algorithm stages
    """

    @staticmethod
    def draw_detections(img, detections, masks):
        """
        Draws heatmasks showing where detector was triggered.
        Bounding boxes and contours are drawn over accepted detections.
        """
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
    def draw_stereo_detections(img1, detections1, masks1,
                               img2, detections2, masks2,
                               assignment=None, assign_data=None,
                               cand_errors=None):
        """
        Draws stereo detections side-by-side and draws lines indicating
        assignments (and near-assignments for debugging)
        """
        import textwrap
        BGR_RED = (0, 0, 255)
        BGR_PURPLE = (255, 0, 255)

        accepted_color = BGR_RED
        rejected_color = BGR_PURPLE

        draw1 = DrawHelper.draw_detections(img1, detections1, masks1)
        draw2 = DrawHelper.draw_detections(img2, detections2, masks2)
        stacked = np.hstack([draw1, draw2])

        if assignment is not None and len(cand_errors) > 0:

            # Draw candidate assignments that did not pass the thresholds
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

                stacked = cv2.line(stacked, center1, center2_,
                                   color=rejected_color, lineType=cv2.LINE_AA,
                                   thickness=1)
                text = textwrap.dedent(
                    '''
                    error = {error:.2f}
                    '''
                ).strip().format(error=error)

                stacked = putMultiLineText(stacked, text, org=center2_,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=rejected_color,
                                           thickness=2, lineType=cv2.LINE_AA)

            # Draw accepted assignments
            for (i, j), info in zip(assignment, assign_data):

                center1 = np.array(detections1[i]['oriented_bbox'].center)
                center2 = np.array(detections2[j]['oriented_bbox'].center)

                # Offset center2 to the right image
                center2_ = center2 + [draw1.shape[1], 0]

                center1 = tuple(center1.astype(np.int))
                center2_ = tuple(center2_.astype(np.int))

                stacked = cv2.line(stacked, center1, center2_,
                                   color=accepted_color, lineType=cv2.LINE_AA,
                                   thickness=2)

                text = textwrap.dedent(
                    '''
                    len = {fishlen:.2f}mm
                    error = {error:.2f}
                    range = {range:.2f}mm
                    '''
                ).strip().format(**info)

                stacked = putMultiLineText(stacked, text, org=center1,
                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1.5, color=accepted_color,
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
    Helper for iterating through a sequence of image frames.
    This will be replaced by KWIVER.
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


def demodata_input(dataset='test'):
    """
    Specifies the input files for testing and demos
    """
    import glob

    if dataset == 'test':
        data_fpath = expanduser('~/data/autoprocess_test_set')
        cal_fpath = join(data_fpath, 'cal_201608.mat')
        img_path1 = join(data_fpath, 'image_data/left')
        img_path2 = join(data_fpath, 'image_data/right')
    elif dataset == 'haul83':
        data_fpath = expanduser('~/data/camtrawl_stereo_sample_data/')
        # cal_fpath = join(data_fpath, 'code/Calib_Results_stereo_1608.mat')
        # cal_fpath = join(data_fpath, 'code/cal_201608.mat')
        cal_fpath = join(data_fpath, '201608_calibration_data/selected/Camtrawl_2016.npz')
        img_path1 = join(data_fpath, 'Haul_83/left')
        img_path2 = join(data_fpath, 'Haul_83/right')
    else:
        assert False, 'bad dataset'

    image_path_list1 = sorted(glob.glob(join(img_path1, '*.jpg')))
    image_path_list2 = sorted(glob.glob(join(img_path2, '*.jpg')))
    assert len(image_path_list1) == len(image_path_list2)
    return image_path_list1, image_path_list2, cal_fpath


def demodata_detections(dataset='haul83', target_step='detect', target_frame_num=7):
    """
    Helper for doctests. Gets test data at different points in the pipeline.
    """
    # <ipython hacks>
    if 'target_step' not in vars():
        target_step = 'detect'
    if 'target_frame_num' not in vars():
        target_frame_num = 7
    # </ipython hacks>
    image_path_list1, image_path_list2, cal_fpath = demodata_input(dataset=dataset)

    cal = StereoCalibration.from_file(cal_fpath)

    detector1 = GMMForegroundObjectDetector()
    detector2 = GMMForegroundObjectDetector()
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

        detections1 = detector1.detect(img1)
        detections2 = detector2.detect(img2)
        masks1 = detector1._masks
        masks2 = detector2._masks

        n_detect1, n_detect2 = len(detections1), len(detections2)
        print('frame_num, (n_detect1, n_detect2) = {} ({}, {})'.format(
            frame_num, n_detect1, n_detect2))

        if frame_num == target_frame_num:
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2)
            dpath = ub.ensuredir('out')
            cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_id), stacked)
            break

    return detections1, detections2, cal


def demo():
    """
    Runs the algorithm end-to-end.
    """
    # dataset = 'test'
    dataset = 'haul83'

    image_path_list1, image_path_list2, cal_fpath = demodata_input(dataset=dataset)

    dpath = ub.ensuredir('out_{}'.format(dataset))

    # ----
    # Choose parameter configurations
    # ----

    # Use GMM based model
    gmm_params = {
        'n_training_frames': 9999,
        # 'gmm_thresh': 20,
        'gmm_thresh': 30,
        'min_size': 800,
        'edge_trim': [40, 40],
        'n_startup_frames': 3,
        'factor': 2,
        'smooth_ksize': None,
        # 'smooth_ksize': (3, 3),
        # 'smooth_ksize': (10, 10),  # wrt original image size
    }
    triangulate_params = {
        'max_err': [6, 14],
        # 'max_err': [200, 200],
    }
    stride = 2

    DRAWING = 0

    # ----
    # Initialize algorithms
    # ----

    detector1 = GMMForegroundObjectDetector(**gmm_params)
    detector2 = GMMForegroundObjectDetector(**gmm_params)
    dfilter1 = FishDetectionFilter(**gmm_params)
    dfilter2 = FishDetectionFilter(**gmm_params)
    triangulator = FishStereoTriangulationAssignment(**triangulate_params)

    import pprint
    print('dataset = {!r}'.format(dataset))
    print('Detector1 Config: ' + pprint.pformat(detector1.config, indent=4))
    print('Detector2 Config: ' + pprint.pformat(detector2.config, indent=4))
    print('Triangulate Config: ' + pprint.pformat(triangulator.config, indent=4))
    pprint.pformat(detector2.config)

    cal = StereoCalibration.from_file(cal_fpath)
    stream1 = FrameStream(image_path_list1, stride=stride)
    stream2 = FrameStream(image_path_list2, stride=stride)

    # ----
    # Run the algorithm
    # ----

    n_total = 0
    all_errors = []
    all_lengths = []

    prog = ub.ProgIter(enumerate(zip(stream1, stream2)),
                       clearline=True,
                       length=len(stream1) // stride,
                       adjust=False)
    _iter = prog

    print('begin iteration')
    for frame_num, ((frame_id1, img1), (frame_id2, img2)) in _iter:
        assert frame_id1 == frame_id2
        frame_id = frame_id1

        detections1_ = list(detector1.detect(img1))
        detections2_ = list(detector2.detect(img2))
        masks1 = detector1._masks
        masks2 = detector2._masks

        dsize1 = tuple(img1.shape[:2][::-1])
        dsize2 = tuple(img2.shape[:2][::-1])

        detections1 = list(dfilter1.filter_detections(detections1_, dsize1))
        detections2 = list(dfilter2.filter_detections(detections2_, dsize2))

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

        if DRAWING:
            stacked = DrawHelper.draw_stereo_detections(img1, detections1, masks1,
                                                        img2, detections2, masks2,
                                                        assignment, assign_data,
                                                        cand_errors)
            stacked = cv2.putText(stacked,
                                  text='frame {}'.format(frame_id),
                                  org=(10, 50),
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=1, color=(255, 0, 0),
                                  thickness=2, lineType=cv2.LINE_AA)
            cv2.imwrite(dpath + '/mask{}_draw.png'.format(frame_id), stacked)

    all_errors = np.array(all_errors)
    all_lengths = np.array(all_lengths)
    print('n_total = {!r}'.format(n_total))
    print('ave_error = {:.2f} +- {:.2f}'.format(all_errors.mean(), all_errors.std()))
    print('ave_lengths = {:.2f} +- {:.2f} '.format(all_lengths.mean(), all_lengths.std()))


if __name__ == '__main__':
    r"""
    CommandLine:

        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh
        export SPROKIT_PYTHON_MODULES=camtrawl_processes:kwiver.processes:viame.processes
        export PYTHONPATH=$(pwd):$PYTHONPATH

        python ~/code/VIAME/plugins/camtrawl/python/camtrawl_algos.py

        ffmpeg -y -f image2 -i out_haul83/%*.png -vcodec mpeg4 -vf "setpts=10*PTS" haul83-results.avi
    """
    demo()
    # import utool as ut
    # ut.dump_profile_text()
    # import pytest
    # pytest.main([__file__, '--doctest-modules'])
