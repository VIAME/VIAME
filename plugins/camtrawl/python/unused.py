import numpy as np
import cv2


def commented_blocks():
    """
    self.background_model = cv2.createBackgroundSubtractorKNN(
        history=self.config['n_training_frames'],
        dist2Threshold=50 ** 2,
        detectShadows=False
    )

    # Grabcut didn't work that well
    if False and self.n_iters > 5:
        # Refine detections with grabcut
        mask = np.zeros(mask.shape, dtype=mask.dtype)
        for detection in detections:
            print('RUNNING GC')
            cc = detection['cc'].astype(np.uint8) * 255
            cc = grabcut(img_, cc)
            mask[cc > 0] = 255
        detections = list(self.masked_detect(mask))
        if return_info:
            masks['cut'] = mask.copy()
    """
    pass


def normalizePixel(pts, fc, cc, kc, alpha_c):
    """
    Alternative to cv2.undistortPoints. The main difference is that this
    runs iterative distortion componsation for 20 iters instead of 5.

    Ultimately, it doesn't make much difference, use opencv instead because
    its faster.
    """
    x_distort = np.array([(pts[0, :] - cc[0]) / fc[0], (pts[1, :] - cc[1]) / fc[1]])
    x_distort[0, :] = x_distort[0, :] - alpha_c * x_distort[1, :]
    if not np.linalg.norm(kc) == 0:
        xn = compDistortion(x_distort, kc)
    else:
        xn = x_distort
    return xn


def compDistortion(xd, k):
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


def grabcut(bgr_img, prior_mask, binary=True, num_iters=5):
    """
    Baseline opencv segmentation algorithm based on graph-cuts.

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
            >>> from camtrawl_algos import *
            >>> #
            >>> from matplotlib import pyplot as plt
            >>> image_path_list1, _, _ = demodata_input(dataset='haul83')
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
        >>> from camtrawl_algos import *
        >>> self, img = demodata_detections(target_step='detect', target_frame_num=7)
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

