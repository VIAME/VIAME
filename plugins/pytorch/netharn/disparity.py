import numpy as np
import kwimage


def compute_disparity(img_left, img_right, disp_range=(0, 240), block_size=11,
                      scale=1.0):
    """
    Compute disparity for a fixed disparity range using SGBM

    The image is returned as floating point disparity values

    References:
        https://kwgitlab.kitware.com/matt.leotta/habcam_stereo/blob/master/disparity.py#L63

    Ignore:
        import ndsampler
        import kwcoco
        fpath = ub.expandpath('$HOME/raid/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_train.mscoco.json')
        dset = kwcoco.CocoDataset(fpath)
        from .detect_fit import DetectFitConfig
        config = DetectFitConfig()
        sampler = ndsampler.CocoSampler(dset, workdir=config['workdir'])
        self = cls(sampler, augment='complex', window_dims=window_dims)


    Ignore:
        >>> img = kwimage.imread('/home/joncrall/raid/data/noaa/2015_Habcam_photos/201503.20150525.101948139.574375.png')
        >>> img = kwimage.imread('/home/joncrall/raid/data/noaa/2015_Habcam_photos/201503.20150517.091650452.40950.png')
        >>> img = kwimage.imread('/home/joncrall/raid/data/noaa/2015_Habcam_photos/201503.20150612.164607433.120500.png')
        >>> img_left = img[:, :img.shape[1] // 2]
        >>> img_right = img[:, img.shape[1] // 2:]
        >>> disp_im = compute_disparity(img_left, img_right, scale=0.5)
        >>> disp_im = multipass_disparity(img_left, img_right, scale=0.5, as01=True)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> kwplot.imshow(img_left, pnum=(2, 2, 1))
        >>> kwplot.imshow(img_right, pnum=(2, 2, 2))
        >>> isvalid = disp_im > 0
        >>> mn = disp_im[isvalid].min()
        >>> mx = disp_im[isvalid].max()
        >>> canvas = (disp_im - mn) / (mx - mn)
        >>> canvas[~isvalid] = 0
        >>> kwplot.imshow(canvas, pnum=(2, 1, 2))
        >>> kwplot.show_if_requested()

    Example:
        >>> img_left = kwimage.grab_test_image('tsukuba_l')
        >>> img_right = kwimage.grab_test_image('tsukuba_r')
        >>> disp_im = compute_disparity(img_left, img_right)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> kwplot.imshow(img_left, pnum=(2, 2, 1))
        >>> kwplot.imshow(img_right, pnum=(2, 2, 2))
        >>> disp_im[disp_im < 0] = 0
        >>> kwplot.imshow(disp_im, pnum=(2, 1, 2))
        >>> kwplot.show_if_requested()
    """
    import cv2
    orig_size = img_left.shape[0:2][::-1]
    if scale != 1.0:
        img_left = kwimage.imresize(img_left, scale=scale)
        img_right = kwimage.imresize(img_right, scale=scale)

    min_disp = int(disp_range[0])
    num_disp = int(disp_range[1] - disp_range[0])
    # num_disp must be a multiple of 16
    num_disp = ((num_disp + 15) // 16) * 16

    disp_alg = cv2.StereoSGBM_create(numDisparities=num_disp,
                                     minDisparity=min_disp,
                                     uniquenessRatio=10,
                                     blockSize=block_size,
                                     speckleWindowSize=0,
                                     speckleRange=0,
                                     P1=8 * block_size**2,
                                     P2=32 * block_size**2)

    disp_int = disp_alg.compute(img_left, img_right)

    # max_disp = 16 * disp_range[1]
    disp_float = disp_int.astype(np.float32) / 16
    # disp_float[disp_float < 0] = -1

    if scale != 1.0:
        disp_float = kwimage.imresize(disp_float, dsize=orig_size,
                                      interpolation='nearest')
    return disp_float


def multipass_disparity(img_left, img_right, outlier_percent=3,
                        range_pad_percent=10, scale=1.0, as01=False):
    """
    Compute dispartity in two passes

    References:
        https://kwgitlab.kitware.com/matt.leotta/habcam_stereo/blob/master/disparity.py#L63

    The first pass obtains a robust estimate of the disparity range
    The second pass limits the search to the estimated range for better
    coverage.

    The outlier_percent variable controls which percentange of extreme
    values to ignore when computing the range after the first pass

    The range_pad_percent variable controls by what percentage to expand
    the range by padding on both the low and high ends to account for
    inlier extreme values that were falsely rejected

    Example:
        >>> img_left = kwimage.grab_test_image('tsukuba_l')
        >>> img_right = kwimage.grab_test_image('tsukuba_r')
        >>> scale = 1.0
        >>> disp_im = multipass_disparity(img_left, img_right, scale=1.0)
        >>> disp_im = multipass_disparity(img_left, img_right, scale=0.5)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> kwplot.imshow(img_left, pnum=(2, 2, 1))
        >>> kwplot.imshow(img_right, pnum=(2, 2, 2))
        >>> disp_im = np.nan_to_num(disp_im)
        >>> kwplot.imshow(disp_im, pnum=(2, 1, 2))
        >>> kwplot.show_if_requested()
    """
    orig_size = img_left.shape[0:2][::-1]
    if scale != 1.0:
        img_left = kwimage.imresize(img_left, scale=scale)
        img_right = kwimage.imresize(img_right, scale=scale)

    # first pass - search the whole range
    disp_img = compute_disparity(img_left, img_right)

    # ignore pixels near the boarder
    border = 20
    disp_core = disp_img[border:-border, border:-border]

    # get a mask of valid disparity pixels
    valid = disp_core >= 0

    # compute a robust range from the valid pixels
    valid_data = disp_core[valid]
    if valid_data.size == 0:
        valid_data = np.array([0, 0, 0, 1, 1, 1])  # hack
    low = np.percentile(valid_data, outlier_percent / 2)
    high = np.percentile(valid_data, 100 - outlier_percent / 2)
    pad = (high - low) * range_pad_percent / 100.0
    low -= pad
    high += pad

    # second pass - limit the search range
    disp_img = compute_disparity(img_left, img_right, (low, high))
    valid = disp_img >= low

    disp_img[~valid] = np.nan

    if as01:
        # normalized value for nets
        disp_img = (disp_img - np.floor(low)) / (np.ceil(high) - np.floor(low))
        disp_img = np.nan_to_num(disp_img)
        mx = disp_img.max()
        # hack
        if mx > 1:
            disp_img = disp_img / mx

    if scale != 1.0:
        disp_img = kwimage.imresize(disp_img, dsize=orig_size,
                                    interpolation='nearest')
    return disp_img


def compute_disparity_old(imgL, imgR, scale=0.5):
    import cv2
    imgL1 = kwimage.imresize(imgL, scale=scale)
    imgR1 = kwimage.imresize(imgR, scale=scale)
    disp_alg = cv2.StereoSGBM_create(numDisparities=16, minDisparity=0,
                                     uniquenessRatio=5, blockSize=15,
                                     speckleWindowSize=50, speckleRange=2,
                                     P1=500, P2=2000, disp12MaxDiff=1000,
                                     mode=cv2.STEREO_SGBM_MODE_HH)
    disparity = disp_alg.compute(
        kwimage.convert_colorspace(imgL1, 'rgb', 'gray'),
        kwimage.convert_colorspace(imgR1, 'rgb', 'gray')
    )
    disparity = disparity - disparity.min()
    disparity = disparity / disparity.max()

    full_dsize = tuple(map(int, imgL.shape[0:2][::-1]))
    disparity = kwimage.imresize(disparity, dsize=full_dsize)
    return disparity
