#!/usr/bin/env python

"""ckwg +29
 * Copyright 2015-2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

Calibrate cameras from a video of a chessboard


"""

import numpy as np
import cv2
import os.path
import glob
import operator

from optparse import OptionParser


def make_object_points(grid_size=(6,5)):
    """construct the array of object points for camera calibration"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)
    return objp


def detect_grid_image(image, grid_size=(6,5)):
    """Detect a grid in a grayscale image"""
    min_len = min(image.shape)
    scale = 1.0
    while scale*min_len > 1000:
        scale /= 2.0
    print("scale = ", scale)


    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find the chess board corners
    flags =  cv2.CALIB_CB_ADAPTIVE_THRESH
    if scale < 1.0:
        small = cv2.resize(image, (0,0), fx=scale, fy=scale)
        ret, corners = cv2.findChessboardCorners(small, grid_size, flags=flags)
        if ret == True:
            cv2.cornerSubPix(small, corners, (11,11), (-1,-1), criteria)
            corners /= scale
    else:
        ret, corners = cv2.findChessboardCorners(image, grid_size, flags=flags)

    if ret == True:
        # refine the location of the corners
        cv2.cornerSubPix(image, corners, (11,11), (-1,-1), criteria)
        return corners
    else:
        return None


def video_frames(video_file, frame_step=1):
    frame_number = 0
    vf = cv2.VideoCapture(video_file)
    if vf.isOpened():
        print("opened video")
        while(vf.isOpened()):
            ret, frame = vf.read()
            if not ret:
                vf.release()
                break
            frame_number += 1
            yield frame, frame_number
        while(vf.isOpened() and frame_number % frame_step != 0):
            ret = vf.grab()
            if not ret:
                vf.release()
            frame_number += 1
    else:
        print("trying file glob",video_file, glob.glob(video_file))
        files = list(enumerate(sorted(glob.glob(video_file))))
        for n, f in files[::frame_step]:
            print(n,f)
            frame = cv2.imread(f)
            yield frame, n
    vf.release()


def detect_grid_video(video_file, grid_size=(6,5), frame_step=1, gui=False, bayer=False):
    """Detect a grid in each frame of video"""

    # Dicts to store corner points from all the images.
    left_data = {}
    right_data = {}

    print("video: ",video_file)
    for frame, frame_number in video_frames(video_file, frame_step):

        left_img = frame[:, 0:frame.shape[1] // 2]
        right_img = frame[:, frame.shape[1] // 2:]
        if bayer:
            left_gray = cv2.cvtColor(left_img[:,:,0], cv2.COLOR_BayerBG2GRAY)
            right_gray = cv2.cvtColor(right_img[:,:,0], cv2.COLOR_BayerBG2GRAY)
        else:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        img_shape = left_gray.shape[::-1]

        left_corners = detect_grid_image(left_gray, grid_size)
        right_corners = detect_grid_image(right_gray, grid_size)

        # If found, add object points, image points (after refining them)
        if left_corners is not None:
            print("found checkerboard in frame %d left" % frame_number)
            left_data[frame_number] = left_corners
        if right_corners is not None:
            print("found checkerboard in frame %d right" % frame_number)
            right_data[frame_number] = right_corners
        if left_corners is None and right_corners is None:
            print("checkerboard not found in %d" % frame_number)


        # Draw and display the corners
        if gui:
            if bayer:
                color_frame = cv2.cvtColor(frame[:,:,0], cv2.COLOR_BayerBG2BGR)
            else:
                color_frame = frame
            if left_corners is not None:
                cv2.drawChessboardCorners(color_frame, grid_size, left_corners, True)
            if right_corners is not None:
                offset = np.repeat(np.array([[[img_shape[0], 0]]],
                                            dtype=right_corners.dtype),
                                   right_corners.shape[0], axis=0)
                shift_right_corners = right_corners + offset
                cv2.drawChessboardCorners(color_frame, grid_size,
                                          shift_right_corners, True)
            cv2.imshow('img',color_frame)
            cv2.waitKey(-1)
    print("done")
    if gui:
        cv2.destroyAllWindows()

    return img_shape, left_data, right_data


def save_calibration(outfile, cal_result):
    ret, mtx, dist, rvecs, tvecs = cal_result

    print("error ", ret)
    print("K = ", mtx.get())
    print("distortion ", dist.get())

    with open(outfile,'w') as f_handle:
        np.savetxt(f_handle, mtx.get())
        f_handle.write("\n")
        np.savetxt(f_handle, dist.get())
        f_handle.write("\nError: %g" % ret)


def evaluate_error(imgpoints, objp, frames, cal_result):
    ret, mtx, dist, rvecs, tvecs = cal_result
    new_frames = []
    new_imgpoints = []
    new_objpoints = []
    frame_errors = {}
    for c, r, t, n in zip(imgpoints, rvecs, tvecs, frames):
        proj_pts, _ = cv2.projectPoints(objp, r, t, mtx, dist)
        frame_errors[n] = np.sqrt(np.mean((proj_pts.get() - c)**2))
        if frame_errors[n] < 1.0:
            new_frames.append(n)
            new_imgpoints.append(c)
            new_objpoints.append(objp)

    for n,v in sorted(frame_errors.items(), key=operator.itemgetter(1)):
        print(v, n)

    return new_frames, new_imgpoints, new_objpoints


def calibrate_single_camera(data, object_points, img_shape):

    objpoints = [object_points] * len(data)
    frames = list(data.keys())
    imgpoints = list(data.values())

    flags = 0
    K = np.matrix([[1000, 0 , img_shape[1]/2],[0, 1000, img_shape[0]/2],[0, 0, 1]])
    d = np.matrix([0,0,0,0])
    cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
    print("initial calibration error: ", cal_result[0])

    # per frame analysis
    #frames, imgpoints, objpoints = evaluate_error(imgpoints, object_points, frames, cal_result)

    ret, mtx, dist, rvecs, tvecs = cal_result
    aspect_ratio = mtx.get()[0,0] / mtx.get()[1,1]
    print("aspect ratio: ",aspect_ratio)
    if 1.0 - min(aspect_ratio, 1.0/aspect_ratio) < 0.01:
        print("fixing aspect ratio at 1.0")
        flags += cv2.CALIB_FIX_ASPECT_RATIO
        cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
        ret, mtx, dist, rvecs, tvecs = cal_result
        print("Fixed aspect ratio error: ", cal_result[0])

    pp = np.array([mtx.get()[0,2], mtx.get()[1,2]])
    print("principal point: ",pp)
    rel_pp_diff = (pp - np.array(img_shape)/2) / np.array(img_shape)
    print("rel_pp_diff", max(abs(rel_pp_diff)))
    if max(abs(rel_pp_diff)) < 0.05:
        print("fixed principal point to image center")
        flags += cv2.CALIB_FIX_PRINCIPAL_POINT
        cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
        print("Fixed principal point error: ", cal_result[0])

    # set a threshold 25% more than the baseline error
    error_threshold = 1.25 * cal_result[0]

    last_result = (cal_result, flags)

    # Ignore tangential distortion
    flags += cv2.CALIB_ZERO_TANGENT_DIST
    cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
    print("No tangential error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K3
    flags += cv2.CALIB_FIX_K3
    cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
    print("No K3 error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K2
    flags += cv2.CALIB_FIX_K2
    cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
    print("No K2 error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    last_result = (cal_result, flags)

    # Ignore K1
    flags += cv2.CALIB_FIX_K1
    cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
    print("No distortion error: ", cal_result[0])
    if cal_result[0] > error_threshold:
        return last_result
    return (cal_result, flags)


def main():
    usage = "usage: %prog [options] video_file\n\n"
    usage += "  Estimate calibration from video.\n"
    parser = OptionParser(usage=usage)

    parser.add_option("-b", "--bayer", default=False,
                      action="store_true", dest="bayer",
                      help="input images are Bayer patterned")
    parser.add_option("-c", "--corners-file", type='string', default=None,
                      action="store", dest="corners_file",
                      help="width of the grid to detect")
    parser.add_option("-q", "--square-size", type='float', default=85,
                      action="store", dest="square_size",
                      help="width of a single calibration square (mm)")
    parser.add_option("-x", "--grid-x", type='int', default=6,
                      action="store", dest="grid_x",
                      help="width of the grid to detect")
    parser.add_option("-y", "--grid-y", type='int', default=5,
                      action="store", dest="grid_y",
                      help="height of the grid to detect")
    parser.add_option("-s", "--frame-step", type='int', default=1,
                      action="store", dest="frame_step",
                      help="number of frames to step between each detection")
    parser.add_option("-g", "--gui", default=False,
                      action="store_true", dest="gui",
                      help="visualize the results in a GUI")

    (options, args) = parser.parse_args()

    video_file = args[0]
    grid_size = (options.grid_x, options.grid_y)

    img_shape = None
    if options.corners_file:
        if os.path.exists(options.corners_file):
            data = np.load(options.corners_file, allow_pickle=True)
            img_shape = tuple(data["img_shape"])
            left_data = data["left_data"].item()
            right_data = data["right_data"].item()

    if img_shape == None:
        img_shape, left_data, right_data = detect_grid_video(video_file, grid_size,
                                                             options.frame_step,
                                                             options.gui, options.bayer)
        if options.corners_file:
            np.savez(options.corners_file, img_shape=img_shape,
                     left_data=left_data, right_data=right_data)

    print("computing calibration")
    objp = make_object_points(grid_size) * options.square_size
    (_, K_left, dist_left, _, _), _ = calibrate_single_camera(left_data, objp, img_shape)
    (_, K_right, dist_right, _, _), _ = calibrate_single_camera(right_data, objp, img_shape)

    # write the intrinsics file
    fs = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_WRITE)
    if (fs.isOpened()):
        fs.write("M1", K_left.get())
        fs.write("D1", dist_left.get())
        fs.write("M2", K_right.get())
        fs.write("D2", dist_right.get())
    fs.release()

    # find frames that detected the target in both left and right views
    frames = set(left_data.keys()).intersection(set(right_data.keys()))
    left_points = [left_data[f] for f in frames]
    right_points = [right_data[f] for f in frames]
    objpoints = [objp] * len(frames)
    ret = cv2.stereoCalibrate(objpoints, left_points, right_points,
                              K_left, dist_left, K_right, dist_right, img_shape,
                              flags=cv2.CALIB_FIX_INTRINSIC)
    R, T = ret[5:7]
    ret2 = cv2.stereoRectify(K_left, dist_left, K_right, dist_right, img_shape, R, T,
                             flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    R1, R2, P1, P2, Q = ret2[:5]

    fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_WRITE)
    if (fs.isOpened()):
        fs.write("R", R.get())
        fs.write("T", T.get())
        fs.write("R1", R1.get())
        fs.write("R2", R2.get())
        fs.write("P1", P1.get())
        fs.write("P2", P2.get())
        fs.write("Q", Q.get())
    fs.release()


if __name__ == "__main__":
    main()

