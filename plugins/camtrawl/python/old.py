    def find_match(self, detections1, detections2, cal, dsize):
        for det1, det2 in it.product(detections1, detections2):
            print('----')

            box_points1 = det1['box_points']
            box_points2 = det2['box_points']

            # pts1 = box_points1[[0, 2]]
            # pts2 = box_points2[[0, 2]]

            pts1 = np.vstack([box_points1, det1['oriented_bbox'].center])
            pts2 = np.vstack([box_points2, det2['oriented_bbox'].center])

            stereo = StereoCameras(cal)

            # pts = box_points1
            # intrinsic = cal['intrinsic_left']
            # Normalize the image projection according to the intrinsic parameters of the left and right cameras
            # pts1_norm = stereo.normalize_pixel(pts1, 'left')
            # pts2_norm = stereo.normalize_pixel(pts2, 'right')

            K1, K2 = stereo.camera_intrinsic_matrices()
            kc1, kc2 = stereo.camera_distortions()

            KL, KR = K1, K2
            kcL, kcR = kc1, kc2

            R = np.diag(stereo.extrinsic['om'])
            T = stereo.extrinsic['T'][:, None]

            # undistort_img1 = cv2.undistort(img1, K1, kc1)
            # cv2.imwrite('img1_distort.png', img1)
            # cv2.imwrite('img1_undistort.png', undistort_img1)

            unpts1 = cv2.undistortPoints(pts1[:, None, :], K1, kc1)[:, 0, :]
            unpts2 = cv2.undistortPoints(pts2[:, None, :], K2, kc2)[:, 0, :]

            rectified = cv2.stereoRectify(K1, kc1, K2, kc2, dsize, R, T)
            (R1, R2, P1, P2, Q, validPixROI1, validPixROI2) = rectified
            print('validPixROI1 = {!r}'.format(validPixROI1))
            print('validPixROI2 = {!r}'.format(validPixROI2))

            # In [176]: print(ut.repr2(P2, precision=3))
            # np.array([[  1.018e+03,   0.000e+00,   6.354e+02,   0.000e+00],
            #           [  0.000e+00,   1.018e+03,   7.804e+02,   2.131e+05],
            #           [  0.000e+00,   0.000e+00,   1.000e+00,   0.000e+00]])

            # In [177]: print(ut.repr2(P1, precision=3))
            # np.array([[  1.018e+03,   0.000e+00,   6.354e+02,   0.000e+00],
            #           [  0.000e+00,   1.018e+03,   7.804e+02,   0.000e+00],
            #           [  0.000e+00,   0.000e+00,   1.000e+00,   0.000e+00]])

            world_pts_homog = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).T
            world_pts = from_homog(world_pts_homog)

            # xys = world_pts

            rvec1 = cv2.Rodrigues(R1)[0]
            rvec2 = cv2.Rodrigues(R2)[0]

            # list(map(cv2.convertPointsFromHomogeneous, world_pts_homog.T))
            # world_pts = (world_pts_homog[0:3] / world_pts_homog[3][None, :])

            # world_pts = stereo.triangulate(pts1, pts2)

            proj_pts1 = cv2.projectPoints(world_pts, rvec1, np.zeros((1, 3)), K1, kc1)[0][:, 0, :]
            proj_pts2 = cv2.projectPoints(world_pts, rvec2, T, K2, kc2)[0][:, 0, :]

            err1 = ((proj_pts1 - pts1) ** 2).sum(axis=1)
            print('err1 = {!r}'.format(err1))
            err2 = ((proj_pts2 - pts2) ** 2).sum(axis=1)
            print('err2 = {!r}'.format(err2))
            print('----')

            # proj_pts_h = P1.dot(to_homog(world_pts).T).T
            # proj_pts = from_homog(proj_pt_h)
            om = cal['extrinsic']['om']
            R = cv2.Rodrigues(om)

            np.linalg.norm(pts1[0] - pts1[1])
            np.linalg.norm(pts2[0] - pts2[1])


@tmp_sprokit_register_process(name='stereo_calibration_camera_reader', doc='preliminatry fish detection')
class CamtrawlStereoCalibrationReaderProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.

    Developer:
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/VIAME/plugins/camtrawl/python')
        >>> from camtrawl_processes import *
        >>> conf = config.empty_config()
        >>> #conf = vital.config_block.ConfigBlock()  # FIXME: should work with vital config
        >>> self = CamtrawlStereoCalibrationReaderProcess(conf)
    """
    # ----------------------------------------------
    def __init__(self, conf):
        log.debug(' ----- init ' + self.__class__.__name__)
        KwiverProcess.__init__(self, conf)

        cal_fpath = expanduser('~/data/autoprocess_test_set/cal_201608.mat')
        default_params = [
            ctalgo.ParamInfo('cal_fpath', default=cal_fpath, doc=(
                'path to a file holding stereo calibration data')),
        ]
        camtrawl_setup_config(self, default_params)

        self.add_port_trait('camera' + '1', 'camera', 'Left camera calibration')
        self.add_port_trait('camera' + '2', 'camera', 'Right camera calibration')

        #  declare our input port ( port-name,flags)
        optional = process.PortFlags()
        self.declare_output_port_using_trait('camera' + '1', optional)
        self.declare_output_port_using_trait('camera' + '2', optional)

        # State used to cache camera loading
        self.cameras = None

    # ----------------------------------------------
    def _configure(self):
        log.debug(' ----- configure ' + self.__class__.__name__)
        config = tmp_smart_cast_config(self)
        print('camera config = {}'.format(ub.repr2(config, nl=2)))
        self._base_configure()

    def load_calibrations(self, cal_fpath):
        cal = ctalgo.StereoCalibration.from_file(cal_fpath)
        def _to_vital(cam_dict):
            from vital.types import camera
            extrinsic = cam_dict['extrinsic']
            intrinsic = cam_dict['intrinsic']
            fx, fy = intrinsic['fc']
            aspect_ratio = fx / fy

            vital_intrinsics = camera.CameraIntrinsics(
                focal_length=fx,
                principle_point=intrinsic['cc'],
                aspect_ratio=aspect_ratio,
                skew=intrinsic['alpha_c'],
                dist_coeffs=intrinsic['kc'],
            )

            tvec = extrinsic['T']
            rvec = extrinsic['om']

            vital_camera = camera.Camera(
                center=tvec,
                rotation=camera.Rotation.from_rodrigues(rvec),
                intrinsics=vital_intrinsics,
            )
            return vital_camera

        camera1 = _to_vital(cal.data['left'])
        camera2 = _to_vital(cal.data['right'])
        return camera1, camera2

    # ----------------------------------------------
    def _step(self):
        log.debug(' ----- step ' + self.__class__.__name__)
        # grab image container from port using traits

        config = tmp_smart_cast_config(self)
        cal_fpath = config['cal_fpath']

        if self.cameras is None:
            self.cameras = self.load_calibrations(cal_fpath)

            camera1, camera2 = self.cameras
        else:
            camera1, camera2 = datum.complete(), datum.complete()
            self.mark_process_as_complete()

        self.push_to_port_using_trait('camera1', camera1)
        self.push_to_port_using_trait('camera2', camera2)

        self._base_step()
