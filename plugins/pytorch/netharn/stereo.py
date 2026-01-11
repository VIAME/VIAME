from os.path import exists
from os.path import join  # NOQA
import cv2
import kwimage
import ubelt as ub
import numpy as np


def _calibrate_single_camera(img_points, object_points, img_dsize):
    K = np.array([[1000, 0, img_dsize[1] / 2],
                  [0, 1000, img_dsize[0] / 2],
                  [0, 0, 1]], dtype=np.float32)
    d = np.array([0, 0, 0, 0], dtype=np.float32)

    objectPoints = [object_points[:, None, :]]
    imgPoints = [img_points[:, None, :]]

    flags = 0
    cal_result = cv2.calibrateCamera(
        objectPoints=objectPoints, imagePoints=imgPoints, imageSize=img_dsize,
        cameraMatrix=K, distCoeffs=d, flags=flags)
    print("initial calibration error: ", cal_result[0])

    # per frame analysis
    #frames, imgPoints, objectPoints = evaluate_error(imgPoints, objectPoints, frames, cal_result)
    ret, mtx, dist, rvecs, tvecs = cal_result
    aspect_ratio = mtx[0, 0] / mtx[1, 1]
    print("aspect ratio: ", aspect_ratio)
    if 1.0 - min(aspect_ratio, 1.0 / aspect_ratio) < 0.01:
        print("fixing aspect ratio at 1.0")
        flags += cv2.CALIB_FIX_ASPECT_RATIO
        cal_result = cv2.calibrateCamera(
            objectPoints, imgPoints, img_dsize, K, d, flags=flags)
        ret, mtx, dist, rvecs, tvecs = cal_result
        print("Fixed aspect ratio error: ", cal_result[0])

    pp = np.array([mtx[0, 2], mtx[1, 2]])
    print("principal point: ", pp)
    rel_pp_diff = (pp - np.array(img_dsize) / 2) / np.array(img_dsize)
    print("rel_pp_diff", max(abs(rel_pp_diff)))
    if max(abs(rel_pp_diff)) < 0.05:
        print("fixed principal point to image center")
        flags += cv2.CALIB_FIX_PRINCIPAL_POINT
        cal_result = cv2.calibrateCamera(
            objectPoints, imgPoints, img_dsize, K, d, flags=flags)
        print("Fixed principal point error: ", cal_result[0])

    # set a threshold 25% more than the baseline error
    error_threshold = 1.25 * cal_result[0]

    last_result = (cal_result, flags)

    # Ignore tangential distortion
    flags += cv2.CALIB_ZERO_TANGENT_DIST
    cal_result = cv2.calibrateCamera(objectPoints, imgPoints, img_dsize, K, d, flags=flags)
    print("No tangential error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K3
    flags += cv2.CALIB_FIX_K3
    cal_result = cv2.calibrateCamera(objectPoints, imgPoints, img_dsize, K, d, flags=flags)
    print("No K3 error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K2
    flags += cv2.CALIB_FIX_K2
    cal_result = cv2.calibrateCamera(objectPoints, imgPoints, img_dsize, K, d, flags=flags)
    print("No K2 error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K1
    flags += cv2.CALIB_FIX_K1
    cal_result = cv2.calibrateCamera(objectPoints, imgPoints, img_dsize, K, d, flags=flags)
    print("No distortion error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    return (cal_result, flags)


def _detect_grid_image(image, grid_dsize):
    """Detect a grid in a grayscale image"""
    min_len = min(image.shape)
    scale = 1.0
    while scale * min_len > 1000:
        scale /= 2.0

    if scale < 1.0:
        small = kwimage.imresize(image, scale=scale)
        return _detect_grid_image(small, grid_dsize)
    else:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Find the chess board corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH
        ret, corners = cv2.findChessboardCorners(image, grid_dsize, flags=flags)
        if ret:
            # refine the location of the corners
            cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
            return corners[:, 0, :]
        else:
            raise Exception('Failed to localize grid')


def _make_object_points(grid_size=(6, 5)):
    """construct the array of object points for camera calibration"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    return objp


def demo_calibrate():
    """
    """
    img_left_fpath = ub.grabdata('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg')
    img_right_fpath = ub.grabdata('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/right01.jpg')
    img_left = kwimage.imread(img_left_fpath)
    img_right = kwimage.imread(img_right_fpath)
    grid_dsize = (6, 9)  # columns x rows
    square_width = 3  # centimeters?

    left_corners = _detect_grid_image(img_left, grid_dsize)
    right_corners = _detect_grid_image(img_right, grid_dsize)
    object_points = _make_object_points(grid_dsize) * square_width

    img_dsize = img_left.shape[0:2][::-1]

    # Intrinstic camera matrix (K) and distortion coefficients (D)
    (_, K1, D1, _, _), _ = _calibrate_single_camera(left_corners, object_points, img_dsize)
    (_, K2, D2, _, _), _ = _calibrate_single_camera(right_corners, object_points, img_dsize)

    objectPoints = [object_points[:, None, :]]
    leftPoints = [left_corners[:, None, :]]
    rightPoints = [right_corners[:, None, :]]
    ret = cv2.stereoCalibrate(objectPoints, leftPoints, rightPoints, K1,
                              D1, K2, D2, img_dsize,
                              flags=cv2.CALIB_FIX_INTRINSIC)
    # extrinsic rotation (R) and translation (T) from the left to right camera
    R, T = ret[5:7]

    # Rectification (R1, R2) matrix (R1 and R2 are homographies, todo: mapping between which spaces?),
    # New camera projection matrix (P1, P2),
    # Disparity-to-depth mapping matrix (Q).
    ret2 = cv2.stereoRectify(K1, D1, K2, D2, img_dsize, R, T,
                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    R1, R2, P1, P2, Q = ret2[:5]

    # NOTE: using cv2.CV_16SC2 is more efficient because it uses a fixed-point
    # encoding of subpixel coordinates, but needs to be decoded to preform the
    # inverse mapping. Using cv2.CV_32FC1 returns a simpler floating-point based
    # representation, which can be directly inverted.
    map11, map12 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_dsize, cv2.CV_16SC2)
    map21, map22 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_dsize, cv2.CV_16SC2)
    map11f, map12f = cv2.convertMaps(map11, map12, cv2.CV_32FC1)
    map21f, map22f = cv2.convertMaps(map21, map22, cv2.CV_32FC1)
    # map11f, map12f = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_dsize, cv2.CV_32FC1)
    # map21f, map22f = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_dsize, cv2.CV_32FC1)

    left_points = np.array(left_corners.tolist() + [
        # hacked extra points
        [0, 0], [3, 3], [5, 5], [10, 10], [15, 15], [19, 19], [31, 31],
        [50, 50], [90, 90], [100, 100], [110, 110],
        [123, 167], [147, 299], [46, 393],
    ])
    right_points = right_corners

    # Map points and images from camera space to rectified space.
    left_rect = cv2.remap(img_left, map11, map12, cv2.INTER_LANCZOS4)
    right_rect = cv2.remap(img_right, map21, map22, cv2.INTER_LANCZOS4)
    left_points_rect = cv2.undistortPoints(left_points, K1, D1, R=R1, P=P1)[:, 0, :]
    right_points_rect = cv2.undistortPoints(right_points, K2, D2, R=R2, P=P2)[:, 0, :]

    if 1:
        def invert_remap(map11f, map12f):
            h, w = map11f.shape[0:2]
            inv_map12f, inv_map11f = np.mgrid[0:h, 0:w].astype(np.float32)
            dx = inv_map11f - map11f
            dy = inv_map12f - map12f
            # inv_map11f + inv_map11f - map11f
            inv_map11f += dx
            inv_map12f += dy
            inv_map11f, inv_map12f
            return inv_map11f, inv_map12f
        # Invert the rectification
        # FIXME: This appears to have an issue
        inv_map11f, inv_map12f = invert_remap(map11f, map12f)
        inv_map21f, inv_map22f = invert_remap(map21f, map22f)
        left_unrect_v1 = cv2.remap(left_rect, inv_map11f, inv_map12f, cv2.INTER_CUBIC)
        right_unrect_v1 = cv2.remap(right_rect, inv_map21f, inv_map22f, cv2.INTER_CUBIC)

    if 1:
        # https://stackoverflow.com/questions/21615298/opencv-distort-back
        # Note negating the distortion coefficients is only a first order approximation
        inv_map11f, inv_map12f = cv2.initUndistortRectifyMap(P1[:, 0:3], -D1, np.linalg.inv(R1), K1, img_dsize, cv2.CV_32FC1)
        inv_map21f, inv_map22f = cv2.initUndistortRectifyMap(P2[:, 0:3], -D2, np.linalg.inv(R2), K2, img_dsize, cv2.CV_32FC1)
        left_unrect_v2 = cv2.remap(left_rect, inv_map11f, inv_map12f, cv2.INTER_CUBIC)
        right_unrect_v2 = cv2.remap(right_rect, inv_map21f, inv_map22f, cv2.INTER_CUBIC)

        if 0:
            inv_map11f, inv_map12f = cv2.initUndistortRectifyMap(P1[:, 0:3], None, np.linalg.inv(R1), K1, img_dsize, cv2.CV_32FC1)
            inv_map21f, inv_map22f = cv2.initUndistortRectifyMap(P2[:, 0:3], None, np.linalg.inv(R2), K2, img_dsize, cv2.CV_32FC1)
            left_unrect_v2 = cv2.remap(left_rect, inv_map11f, inv_map12f, cv2.INTER_CUBIC)
            right_unrect_v2 = cv2.remap(right_rect, inv_map21f, inv_map22f, cv2.INTER_CUBIC)
            left_unrect2 = kwimage.overlay_alpha_layers([
                kwimage.ensure_alpha_channel(left_unrect_v2, 0.65),
                img_left,
            ])
            _, ax7 = kwplot.imshow(left_unrect2, pnum=(1, 1, 1), title='un-rectified V2 (with overlay)')

            import xdev
            D3 = D1.copy()
            for f in xdev.InteractiveIter(np.linspace(-1., 1.), 10):
                print('f = {!r}'.format(f))

                D3[0] = -D1[0] * 0.27
                # D3[2] = f
                D3[4] = f
                inv_map11f, inv_map12f = cv2.initUndistortRectifyMap(P1[:, 0:3], D3, np.linalg.inv(R1), K1, img_dsize, cv2.CV_32FC1)
                left_unrect_v2 = cv2.remap(left_rect, inv_map11f, inv_map12f, cv2.INTER_CUBIC)
                right_unrect_v2 = cv2.remap(right_rect, inv_map21f, inv_map22f, cv2.INTER_CUBIC)
                left_unrect2 = kwimage.overlay_alpha_layers([
                    kwimage.ensure_alpha_channel(left_unrect_v2, 0.65),
                    img_left,
                ])
                _, ax7 = kwplot.imshow(left_unrect2, pnum=(1, 1, 1), title='un-rectified V2 (with overlay)')
                xdev.InteractiveIter.draw()

        # H1_inv = np.linalg.inv(P1[:, 0:3]) @ np.linalg.inv(R1)  # @ np.linalg.inv(P1[:, 0:3])
        # H2_inv = np.linalg.inv(R2)  # @ np.linalg.inv(P2[:, 0:3])
        # left_unrect = cv2.warpPerspective(left_rect, H1_inv, img_dsize)
        # right_unrect = cv2.warpPerspective(right_rect, H2_inv, img_dsize)

    import kwplot
    kwplot.autompl()

    nrows = 4

    kwplot.figure(fnum=1, doclf=True)
    _, ax1 = kwplot.imshow(img_left, pnum=(nrows, 2, 1), title='raw')
    _, ax2 = kwplot.imshow(img_right, pnum=(nrows, 2, 2))
    kwplot.draw_points(left_points, color='red', radius=7, ax=ax1)
    kwplot.draw_points(right_points, color='red', radius=7, ax=ax2)

    _, ax3 = kwplot.imshow(left_rect, pnum=(nrows, 2, 3), title='rectified')
    _, ax4 = kwplot.imshow(right_rect, pnum=(nrows, 2, 4))
    kwplot.draw_points(left_points_rect, color='red', radius=7, ax=ax3)
    kwplot.draw_points(right_points_rect, color='red', radius=7, ax=ax4)

    # v1
    left_unrect2 = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(left_unrect_v1, 0.65),
        img_left,
    ])
    right_unrect2 = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(right_unrect_v1, 0.65),
        img_right,
    ])
    _, ax5 = kwplot.imshow(left_unrect2, pnum=(nrows, 2, 5), title='un-rectified V1 (with overlay)')
    _, ax6 = kwplot.imshow(right_unrect2, pnum=(nrows, 2, 6))

    # V2
    left_unrect2 = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(left_unrect_v2, 0.65),
        img_left,
    ])
    right_unrect2 = kwimage.overlay_alpha_layers([
        kwimage.ensure_alpha_channel(right_unrect_v2, 0.65),
        img_right,
    ])
    _, ax7 = kwplot.imshow(left_unrect2, pnum=(nrows, 2, 7), title='un-rectified V2 (with overlay)')
    _, ax8 = kwplot.imshow(right_unrect2, pnum=(nrows, 2, 8))

    if 1:
        # This seems to work very well
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        rect = cv2.convertPointsToHomogeneous(left_points_rect)[:, 0, :]
        rect = kwimage.warp_points(np.linalg.inv(R1) @ np.linalg.inv(P1[:, 0:3]), rect)
        left_points_unrect, _ = cv2.projectPoints(rect, rvec, tvec, cameraMatrix=K1, distCoeffs=D1)
        left_points_unrect = left_points_unrect.reshape(-1, 2)

        err = left_points_unrect - left_points
        med_err = np.median(err)
        import kwarray
        print('error = ' + ub.repr2(kwarray.stats_dict(err, median=True)))
        kwplot.draw_points(left_points_unrect, color='orange', radius=7, ax=ax5)

        kwplot.draw_points(left_points, color='red', radius=2, ax=ax7)
        kwplot.draw_points(left_points_unrect, color='orange', radius=2, ax=ax7)

    if 0:
        # This works ok, but fails on out-of-bounds points
        left_points_unrect = np.hstack([
            kwimage.subpixel_getvalue(map11, left_points_rect[:, ::-1])[:, None],
            kwimage.subpixel_getvalue(map12, left_points_rect[:, ::-1])[:, None]
        ])
        kwplot.draw_points(left_points_unrect, color='purple', radius=7, ax=ax5)
        err = left_points_unrect - left_points
        med_err = np.median(err)
        print('med_err = {!r}'.format(med_err))



# from viame.pytorch.netharn.metrics.confusion_vectors import DictProxy  # NOQA
try:
    from kwcoco.metrics.util import DictProxy  # NOQA
except ImportError:
    from kwcoco.util.dict_like import DictProxy  # NOQA


class StereoCamera(ub.NiceRepr, DictProxy):
    """
    Helper for single-camera operations with a known rectification.

    References:
        http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html
        http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
        https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
        https://opencv.yahoogroups.narkive.com/guu3Roys/reverse-remap
        https://programtalk.com/vs2/python/8176/opencv-python-blueprints/chapter4/scene3D.py/
        https://stackoverflow.com/questions/21615298/opencv-distort-back
        https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid
        https://stackoverflow.com/questions/51895602/opencv-are-lens-distortion-coefficients-inverted-for-projectpoints
    """
    def __init__(camera):
        camera.proxy = {
            'K': None,  # intrinstics matrix
            'D': None,  # intrinstic distortion coefficients
            'R': None,  # R1 Output 3x3 rectification transform
            'P': None,  # P1 Output 3x4 projection matrix in the new (rectified) coordinate systems.
        }

    def _decompose_intrinsics(camera):
        """
        convert intrinsic matrix to dict

        Notes:
            fc: focal length of the camera
            cc: principle point
            alpha_c: skew
            kc: distortion
        """
        intrin = {}
        fc = intrin['fc'] = np.zeros(2)
        cc = intrin['cc'] = np.zeros(2)
        [[fc[0], alpha_c_fc0, cc[0]],
         [    _,       fc[1], cc[1]],
         [    _,           _,     _]] = camera['K']
        intrin['alpha_c'] = np.array([alpha_c_fc0 / fc[0]]).item()
        intrin['kc'] = camera['D'].ravel()
        return intrin

    def to_vital(camera):
        """
        Convert to a kwiver.vital.types.CameraIntrinsics object

        Ignore:
            camera = camera1
            img_rect = img_rect1
            interpolation = 'nearest'

            intrin = camera._decompose_intrinsics()
            fc = intrin['fc']
            pp = intrin['cc']
            skew = intrin['alpha_c']

            from kwiver.vital import types
            types.CameraIntrinsics()
        """
        intrin = camera._decompose_intrinsics()
        fc = intrin['fc']
        pp = intrin['cc']
        skew = intrin['alpha_c']
        dist_coeffs = intrin['kc']
        focal_length = fc[0]
        aspect_ratio = fc[0] / fc[1]

        from kwiver.vital.types import CameraIntrinsics
        vital_cam = CameraIntrinsics(
            focal_length=focal_length, principal_point=pp,
            aspect_ratio=aspect_ratio, skew=skew, dist_coeffs=dist_coeffs)
        return vital_cam

    def __nice__(camera):
        return ub.repr2(camera.proxy, nl=2)

    def _precache(camera, img_dsize):
        """
        Precomputes distortion and undistortion maps for a given image size.

        In other words for each pixel we need to be able to compute a mapping
        for where it should go, so we need the image size to allocate and
        populate that mapping.
        """
        camera._undistort_rectify_map(img_dsize)
        camera._distort_unrectify_map(img_dsize)
        camera._inv_PR()

    @ub.memoize_method
    def _undistort_rectify_map(camera, img_dsize):
        """
        Cached mapping from raw space -> rectified space
        """
        K, D, R, P = ub.take(camera, ['K', 'D', 'R', 'P'])
        map_x, map_y = cv2.initUndistortRectifyMap(
            K, D, R, P, img_dsize, cv2.CV_16SC2)
        return map_x, map_y

    @ub.memoize_method
    def _distort_unrectify_map(camera, img_dsize):
        """
        Cached mapping from rectified space -> raw space
        """
        w, h = img_dsize
        # Create coordinates for each point in the distorted image
        fw_unrect_xy = np.mgrid[0:w, 0:h].astype(np.float32).T.reshape(-1, 2)
        # Find where each point maps to in the rectified image
        fw_rect_xy = camera.rectify_points(fw_unrect_xy).astype(np.float32)
        inv_map_xf, inv_map_yf = fw_rect_xy.reshape(h, w, 2).transpose(2, 0, 1)
        return inv_map_xf, inv_map_yf

    @ub.memoize_method
    def _inv_PR(camera):
        """
        I dont actually have an intuition for this transform.
        I suppose it unprojects from a rectified camera, and then unrectifies
        back into undistorted normalized coordinates? But Im not sure.
        """
        R, P = camera['R'], camera['P']
        assert np.allclose(P[1:, 3], 0)  # not sure what the last column is
        inv_RP = np.linalg.inv(P[:, 0:3] @ R)
        return inv_RP

    def rectify_image(camera, img_unrect, interpolation='auto'):
        interpolation = kwimage.im_cv2._coerce_interpolation(
            interpolation, default='linear')
        img_dsize = img_unrect.shape[0:2][::-1]
        map_x, map_y = camera._undistort_rectify_map(img_dsize)
        img_rect = cv2.remap(img_unrect, map_x, map_y, cv2.INTER_CUBIC)
        return img_rect

    def unrectify_image(camera, img_rect, interpolation='auto'):
        """
        Warp image from rectified space -> raw space
        """
        interpolation = kwimage.im_cv2._coerce_interpolation(
            interpolation, default='linear')
        img_dsize = img_rect.shape[0:2][::-1]
        inv_map_xf, inv_map_yf = camera._distort_unrectify_map(img_dsize)
        # Use this mapping to invert the rectification transform
        img_unrect = cv2.remap(img_rect, inv_map_xf, inv_map_yf, interpolation)
        return img_unrect

    def rectify_points(camera, points):
        """
        Warp points from raw space -> rectified space
        """
        K, D, R, P = ub.take(camera, ['K', 'D', 'R', 'P'])
        point_rect = cv2.undistortPoints(points, K, D, R=R, P=P)[:, 0, :]
        return point_rect

    def unrectify_points(camera, points_rect):
        """
        Warp points from rectified space -> raw space
        """
        # This seems to work very well
        K, D = camera['K'], camera['D']
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        inv_RP = camera._inv_PR()
        rect = cv2.convertPointsToHomogeneous(points_rect)[:, 0, :]
        rect = kwimage.warp_points(inv_RP, rect)
        points_unrect, _ = cv2.projectPoints(rect, rvec, tvec, cameraMatrix=K, distCoeffs=D)
        points_unrect = points_unrect.reshape(-1, 2)
        return points_unrect


class StereoCalibration():
    """
    Ignore:
        >>> cali = StereoCalibration()
    """
    def __init__(cali):
        cali.cameras = {}
        cali.extrinsics = {}

    def triangulate(cali, pts1, pts2):
        """
        Given two calibrated cameras, and points in each triangulate them

        Example:
            >>> from .stereo import *  # NOQA
            >>> cali = StereoCalibration.demo()
            >>> img1 = cali.img1
            >>> img2 = cali.img2
            >>> #
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> nrows = 1
            >>> _, ax1 = kwplot.imshow(img1, pnum=(nrows, 2, 1), title='raw')
            >>> _, ax2 = kwplot.imshow(img2, pnum=(nrows, 2, 2))

        pts1 = np.array([
            (245.5, 97.0),
            (246.5, 98.0),
            (179.5, 323.0),
            (226.0, 391.0),
            (516.0, 90.0),
        ])
        pts2 = np.array([
            (128., 113.),
            (129., 114.),
            (124., 334.),
            (166., 406.),
            (382., 96.),
        ])

        pts1 = cali.corners1
        pts2 = cali.corners2
        """
        # Move into opencv point format (num x 1 x dim=2)
        pts1_cv = pts1[:, None, :]
        pts2_cv = pts2[:, None, :]

        # Grab camera parameters
        K1 = cali.cameras[1]['K']
        kc1 = cali.cameras[1]['D']

        kc2 = cali.cameras[2]['D']
        K2 = cali.cameras[2]['K']

        # Make extrincic matrices
        rvec1 = np.zeros((3, 1))
        R1 = cv2.Rodrigues(rvec1)[0]
        T1 = np.zeros((3, 1))
        tvec1 = T1

        R2 = cali.extrinsics['R']
        rvec2 = cv2.Rodrigues(R2)[0]
        T2 = cali.extrinsics['T']
        tvec2 = T2

        RT1 = np.hstack([R1, T1])
        RT2 = np.hstack([R2, T2])

        # Undistort points (num x 1 x dim=2)
        # This puts points in "normalized camera coordinates" making them
        # independent of the intrinsic parameters. Moving to world coordinates
        # can now be done using only the RT transform.
        unpts1_cv = cv2.undistortPoints(pts1_cv, K1, kc1)
        unpts2_cv = cv2.undistortPoints(pts2_cv, K2, kc2)

        # note: trinagulatePoints docs say that it wants a 3x4 projection
        # matrix (ie K.dot(RT)), but we only need to use the RT extrinsic
        # matrix because the undistorted points already account for the K
        # intrinsic matrix.
        #
        # Input 2d-points should be (dim=2 x num)
        unpts1_T = unpts1_cv[:, 0, :].T
        unpts2_T = unpts2_cv[:, 0, :].T
        # homog points returned as (dim=4 x num)
        world_pts_homogT = cv2.triangulatePoints(RT1, RT2, unpts1_T, unpts2_T)

        # Remove homogenous coordinate
        # Returns (num x dim=3) world coordinates
        world_pts = kwimage.remove_homog(world_pts_homogT.T)

        # Reproject points
        world_pts_cv = world_pts[:, None, :]
        proj_pts1_cv = cv2.projectPoints(world_pts_cv, rvec1, tvec1, K1, kc1)[0]
        proj_pts2_cv = cv2.projectPoints(world_pts_cv, rvec2, tvec2, K2, kc2)[0]

        # Check error
        err1 = ((proj_pts1_cv - pts1_cv)[:, 0, :] ** 2).sum(axis=1)
        err2 = ((proj_pts2_cv - pts2_cv)[:, 0, :] ** 2).sum(axis=1)
        errors = np.hstack([err1, err2])

        # Get 3d points in each camera's reference frame
        # Note RT1 is the identity and RT are 3x4, so no need for `from_homog`
        # Return points in with shape (N,3)
        pts1_3d = RT1.dot(kwimage.add_homog(world_pts).T).T
        pts2_3d = RT2.dot(kwimage.add_homog(world_pts).T).T

        if False:
            kwplot.plot_points3d(*world_pts.T)
            kwplot.plot_points3d(*pts1_3d.T)
            kwplot.plot_points3d(*pts2_3d.T)

    @classmethod
    def demo(StereoCalibration):
        """
        Example:
            >>> from .stereo import *  # NOQA
            >>> cali = StereoCalibration.demo()
            >>> camera1 = cali.cameras[1]
            >>> camera2 = cali.cameras[2]
            >>> points1 = cali.corners1
            >>> points2 = cali.corners2
            >>> img1 = cali.img1
            >>> img2 = cali.img2
            >>> #
            >>> # Map points and images from camera space to rectified space.
            >>> img_rect1 = camera1.rectify_image(img1)
            >>> img_rect2 = camera2.rectify_image(img2)
            >>> points_rect1 = camera1.rectify_points(points1)
            >>> points_rect2 = camera2.rectify_points(points2)
            >>> #
            >>> # Map them back
            >>> img_unrect1 = camera1.unrectify_image(img_rect1)
            >>> img_unrect2 = camera2.unrectify_image(img_rect2)
            >>> points_unrect1 = camera1.unrectify_points(points_rect1)
            >>> points_unrect2 = camera2.unrectify_points(points_rect2)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> nrows = 3
            >>> _, ax1 = kwplot.imshow(img1, pnum=(nrows, 2, 1), title='raw')
            >>> _, ax2 = kwplot.imshow(img2, pnum=(nrows, 2, 2))
            >>> kwplot.draw_points(points1, color='red', radius=7, ax=ax1)
            >>> kwplot.draw_points(points2, color='red', radius=7, ax=ax2)
            >>> _, ax3 = kwplot.imshow(img_rect1, pnum=(nrows, 2, 3), title='rectified')
            >>> _, ax4 = kwplot.imshow(img_rect2, pnum=(nrows, 2, 4))
            >>> kwplot.draw_points(points_rect1, color='red', radius=7, ax=ax3)
            >>> kwplot.draw_points(points_rect2, color='red', radius=7, ax=ax4)
            >>> canvas_unrect1 = kwimage.overlay_alpha_layers([
            >>>     kwimage.ensure_alpha_channel(img_unrect1, 0.65), img1])
            >>> canvas_unrect2 = kwimage.overlay_alpha_layers([
            >>>     kwimage.ensure_alpha_channel(img_unrect2, 0.65), img2])
            >>> _, ax5 = kwplot.imshow(canvas_unrect1, pnum=(nrows, 2, 5), title='un-rectified (with overlay)')
            >>> _, ax6 = kwplot.imshow(canvas_unrect2, pnum=(nrows, 2, 6))
            >>> kwplot.draw_points(points_unrect1, color='red', radius=7, ax=ax5)
            >>> kwplot.draw_points(points_unrect2, color='red', radius=7, ax=ax6)
        """
        img_left_fpath = ub.grabdata('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg')
        img_right_fpath = ub.grabdata('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/right01.jpg')
        img_left = kwimage.imread(img_left_fpath)
        img_right = kwimage.imread(img_right_fpath)
        grid_dsize = (6, 9)  # columns x rows
        square_width = 3  # centimeters?

        left_corners = _detect_grid_image(img_left, grid_dsize)
        right_corners = _detect_grid_image(img_right, grid_dsize)
        object_points = _make_object_points(grid_dsize) * square_width

        img_dsize = img_left.shape[0:2][::-1]

        # Intrinstic camera matrix (K) and distortion coefficients (D)
        (_, K1, D1, _, _), _ = _calibrate_single_camera(left_corners, object_points, img_dsize)
        (_, K2, D2, _, _), _ = _calibrate_single_camera(right_corners, object_points, img_dsize)

        objectPoints = [object_points[:, None, :]]
        leftPoints = [left_corners[:, None, :]]
        rightPoints = [right_corners[:, None, :]]
        ret = cv2.stereoCalibrate(objectPoints, leftPoints, rightPoints, K1,
                                  D1, K2, D2, img_dsize,
                                  flags=cv2.CALIB_FIX_INTRINSIC)
        # extrinsic rotation (R) and translation (T) from the left to right camera
        R, T = ret[5:7]

        # Rectification (R1, R2) matrix (R1 and R2 are homographies, todo: mapping between which spaces?),
        # New camera projection matrix (P1, P2),
        # Disparity-to-depth mapping matrix (Q).
        ret2 = cv2.stereoRectify(K1, D1, K2, D2, img_dsize, R, T,
                                 flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        R1, R2, P1, P2, Q = ret2[:5]
        cali = StereoCalibration()
        cali.cameras = {
            1: StereoCamera(),
            2: StereoCamera(),
        }
        cali.cameras[1]['K'] = K1
        cali.cameras[1]['D'] = D1
        cali.cameras[1]['R'] = R1
        cali.cameras[1]['P'] = P1

        cali.cameras[2]['K'] = K1
        cali.cameras[2]['D'] = D1
        cali.cameras[2]['R'] = R1
        cali.cameras[2]['P'] = P1

        cali.extrinsics = {
            # Extrinsic Rotation and translation between
            'R': R,
            'T': T,
        }
        cali.corners1 = left_corners
        cali.corners2 = right_corners
        cali.corners_world = object_points
        cali.img1 = img_left
        cali.img2 = img_right
        return cali

    @classmethod
    def from_cv2_yaml(StereoCalibration, intrinsics_fpath, extrinsics_fpath):
        """
        Ignore:
            from .stereo import *  # NOQA
            cali_root = ub.expandpath('~/remote/namek/data/noaa_habcam/extras/calibration_habcam_2019_leotta')
            extrinsics_fpath = join(cali_root, 'extrinsics.yml')
            intrinsics_fpath = join(cali_root, 'intrinsics.yml')
            cali = StereoCalibration.from_cv2_yaml(intrinsics_fpath, extrinsics_fpath)
            camera1 = cali.cameras[1]
            camera2 = cali.cameras[2]

            root = ub.expandpath('~/remote/namek/data/noaa_habcam')
            img1 = kwimage.imread(join(root, '2019_CFARM_P2/images/left/201901.20190629.120403239.112677.cog.tif'))
            img2 = kwimage.imread(join(root, '2019_CFARM_P2/images/right/201901.20190629.120403239.112677.cog.tif'))

            img_rect1 = camera1.rectify_image(img1)
            img_rect2 = camera2.rectify_image(img2)

            from .disparity import multipass_disparity
            disparity_rect1 = multipass_disparity(
                img_rect1, img_rect2, scale=0.5, as01=True)

            img_rect = disparity_rect1

            disparity_unrect1 = camera1.unrectify_image(disparity_rect1)
            img_unrect1 = camera1.unrectify_image(img_rect1)

            import kwplot
            kwplot.autompl()

            kwplot.imshow(img_rect1, pnum=(2, 3, 1), fnum=1)
            kwplot.imshow(img_rect2, pnum=(2, 3, 2), fnum=1)
            kwplot.imshow(disparity_rect1, pnum=(2, 3, 3), fnum=1, cmap='magma')

            kwplot.imshow(img1, pnum=(2, 3, 4), fnum=1)
            kwplot.imshow(img_unrect1, pnum=(2, 3, 5), fnum=1, cmap='magma')
            kwplot.imshow(disparity_unrect1, pnum=(2, 3, 6), fnum=1, cmap='magma')
        """
        import cv2
        assert exists(intrinsics_fpath)
        assert exists(extrinsics_fpath)
        in_fs = cv2.FileStorage(intrinsics_fpath, flags=0)
        ex_fs = cv2.FileStorage(extrinsics_fpath, flags=0)

        cali = StereoCalibration()
        cali.cameras = {
            1: StereoCamera(),
            2: StereoCamera(),
        }
        cali.cameras[1]['K'] = in_fs.getNode("M1").mat()  # intrinstics matrix
        cali.cameras[1]['D'] = in_fs.getNode("D1").mat()  # intrinstic distortion coefficients
        cali.cameras[1]['R'] = ex_fs.getNode("R1").mat()  # R1 Output 3x3 rectification transform
        cali.cameras[1]['P'] = ex_fs.getNode("P1").mat()  # P1 Output 3x4 projection matrix in the new (rectified) coordinate systems.

        cali.cameras[2]['K'] = in_fs.getNode("M2").mat()  # intrinstics matrix
        cali.cameras[2]['D'] = in_fs.getNode("D2").mat()  # intrinstic distortion coefficients
        cali.cameras[2]['R'] = ex_fs.getNode("R2").mat()  # R1 Output 3x3 rectification transform
        cali.cameras[2]['P'] = ex_fs.getNode("P2").mat()  # P1 Output 3x4 projection matrix in the new (rectified) coordinate systems.

        cali.extrinsics = {
            # Extrinsic Rotation and translation between
            'R': ex_fs.getNode("R").mat(),
            'T': ex_fs.getNode("T").mat(),
        }
        return cali


def _notes2(cali):
    """
    gids = 18, 22, 25, 27, 60, 65, 77, 81, 83, 97
    gid = 2182

    img = dset.imgs[gid]
    right_gpath = join(dset.img_root, img['right_cog_name'])
    left_gpath = join(dset.img_root, img['file_name'])
    imgL = kwimage.imread(left_gpath)
    imgR = kwimage.imread(right_gpath)


    import cv2
    M1, D1, M2, D2 = ub.take(cali.intrinsics, [
        'M1', 'D1', 'M2', 'D2'])
    R, T, R1, R2, P1, P2, Q = ub.take(cali.extrinsics, [
        'R', 'T', 'R1', 'R2', 'P1', 'P2', 'Q'])
    K1 = M1
    kc1 = D1
    K2 = M2
    kc2 = D2
    RT = np.hstack([R, T])

    img_dsize = imgL.shape[0:2][::-1]
    map11, map12 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, img_dsize, cv2.CV_16SC2)
    map21, map22 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, img_dsize, cv2.CV_16SC2)

    corners = [
        map11[0, 0],
        map11[0, -1],
        map11[-1, 0],
        map11[-1, -1],
    ]

    left_rect = cv2.remap(imgL, map11, map12, cv2.INTER_CUBIC)
    right_rect = cv2.remap(imgR, map21, map22, cv2.INTER_CUBIC)

    from .disparity import multipass_disparity
    img_disparity = multipass_disparity(
        left_rect, right_rect, scale=0.5, as01=True)

    kwplot.imshow(img_disparity, pnum=(2, 3, 3), fnum=1)
    kwplot.imshow(right_rect, pnum=(2, 3, 2), fnum=1)
    kwplot.imshow(left_rect, pnum=(2, 3, 1), fnum=1)

    annots = dset.annots(gid=gid)
    dets = annots.detections
    self = boxes = dets.data['boxes']
    new_boxes = boxes.warp(R1, homog_mode='divide')
    dets.data['boxes'] = new_boxes.to_xywh()
    dets.draw()

    kwplot.imshow(imgL, pnum=(2, 3, 4), fnum=1)
    annots = dset.annots(gid=gid)
    dets = annots.detections
    dets.draw()

    coords = dets.boxes.to_polygons().data[0].data['exterior']
    matrix = P1
    pts = coords.data

    # Undistort points
    # This puts points in "normalized camera coordinates" making them
    # independent of the intrinsic parameters. Moving to world coordinates
    # can now be done using only the RT transform.
    [[fx, _, cx, _], [_, fy, cy, ty], [_, _, _, tz]] = P1
    pts1 = pts
    pts1_homog = np.hstack([pts1, [[1]] * len(pts1)])
    rectified_pts1_homog = kwimage.warp_points(R1, pts1_homog)
    rectified_pts1 = rectified_pts1_homog[:, 0:2]

    dets.data['boxes']
    import kwimage
    unpts1 = cv2.undistortPoints(pts1[:, None, :], K1, kc1)[:, 0, :]
    unpts1_homog = np.hstack([unpts1, [[1]] * len(unpts1)])
    unpts2_homog = kwimage.warp_points(RT, unpts1_homog)
    kwimage.warp_points(P1[0:3, 0:3], unpts1_homog)
    unpts2_cv = cv2.undistortPoints(pts2_cv, K2, kc2)
    dets.warp(P1)


    # stretch the range of the disparities to [0,255] for display
    disp_img = img_disparity
    img3d = cv2.reprojectImageTo3D(disp_img, Q)
    valid = disp_img > 0
    pts3d = img3d[valid]
    depths = pts3d[:, 2]
    import kwarray
    print(kwarray.stats_dict(depths))

    pip install plottool_ibeis
    pip install vtool_ibeis

    # img_left = kwimage.grab_test_image('tsukuba_l')
    # img_right = kwimage.grab_test_image('tsukuba_r')
    import pyhesaff
    kpts1, desc1 = pyhesaff.detect_feats_in_image(img_left)
    kpts2, desc2 = pyhesaff.detect_feats_in_image(img_right)

    from vtool_ibeis import matching
    annot1 = {'kpts': kpts1, 'vecs': desc1, 'rchip': img_left}
    annot2 = {'kpts': kpts2, 'vecs': desc2, 'rchip': img_right}
    match = matching.PairwiseMatch(annot1, annot2)
    match.assign()

    idx1, idx2 = match.fm.T
    xy1_m = kpts1[idx1, 0:2]
    xy2_m = kpts2[idx2, 0:2]

    # TODO: need to better understand R1, and P1 and what
    # initUndistortRectifyMap does wrt to K and D.
    # cv2.initUndistortRectifyMap(K1, D1, np.linalg.inv(R1), np.linalg.pinv(P1)[0:3], img_dsize, cv2.CV_32FC1)
    # cv2.initUndistortRectifyMap(np.eye(3), None, np.linalg.inv(R1), np.eye(3), img_dsize, cv2.CV_32FC1)

    # Invert rectification?
    # https://groups.google.com/forum/#!topic/pupil-discuss/8eSuYYNEaIQ
    # https://stackoverflow.com/questions/35060164/reverse-undistortrectifymap
    # https://answers.opencv.org/question/129425/difference-between-undistortpoints-and-projectpoints-in-opencv/

    # K1_inv = np.linalg.inv(K1)
    # left_points_unrect = cv2.undistortPoints(left_points_rect, K1, D1, R=R1, P=P1)[:, 0, :]
    # left_points_unrect = cv2.undistortPoints(left_points_rect, P1[:, 0:3], D1, R=R1, P=K1)[:, 0, :]
    # left_points_unrect = cv2.undistortPoints(left_points_rect, P1[:, 0:3], D1, R=None, P=K1)[:, 0, :]
    # M = np.linalg.inv(P1[:, 0:3]) @ R1 @ K1
    # M = K1 @ R1 @ np.linalg.inv(P1[:, 0:3])
    # left_points_unrect = kwimage.warp_points(M, left_points_rect)
    # kwplot.draw_points(left_points_unrect, color='red', radius=7, ax=ax1)
    """
    if 0:
        E, Emask = cv2.findEssentialMat(left_corners, right_corners, K1)  # NOQA
        F, Fmask = cv2.findFundamentalMat(left_corners, right_corners,  # NOQA
                                          cv2.FM_RANSAC, 0.1, 0.99)  # NOQA
        E = K1.T.dot(F).dot(K1)  # NOQA


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/bioharn/stereo.py
    """
    import kwplot
    plt = kwplot.autoplt()
    demo_calibrate()
    plt.show()
