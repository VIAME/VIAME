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

