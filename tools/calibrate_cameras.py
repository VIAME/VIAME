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
import os
import glob
import operator
import json

from optparse import OptionParser


def make_object_points(grid_size=(6,5)):
    """construct the array of object points for camera calibration"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_size[0],0:grid_size[1]].T.reshape(-1,2)
    return objp


def detect_grid_image(image, grid_size=(6,5), max_dim=5000):
    """Detect a grid in a grayscale image"""
    min_len = min(image.shape)
    scale = 1.0
    while scale*min_len > max_dim:
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


DEFAULT_VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}
DEFAULT_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def is_video_file(filepath, video_extensions=None):
    """Check if a file is a video (not an image) by extension"""
    if video_extensions is None:
        video_extensions = DEFAULT_VIDEO_EXTENSIONS
    _, ext = os.path.splitext(filepath.lower())
    return ext in video_extensions


def image_frames(input_path, frame_step=1, image_extensions=None):
    """Yield frames from image file(s) - single file, directory, or glob pattern"""
    if image_extensions is None:
        image_extensions = DEFAULT_IMAGE_EXTENSIONS

    if os.path.isdir(input_path):
        files = []
        for ext in image_extensions:
            pattern = '*' + ext if ext.startswith('.') else '*.' + ext
            files.extend(glob.glob(os.path.join(input_path, pattern)))
            files.extend(glob.glob(os.path.join(input_path, pattern.upper())))
        files = sorted(files)
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        files = sorted(glob.glob(input_path))

    if len(files) == 0:
        raise ValueError(f"No images found. Input '{input_path}' is not a valid "
                         f"directory or glob pattern matching any images.")

    print(f"processing {len(files)} image(s)")
    frames_yielded = 0
    for n, f in enumerate(files):
        if n % frame_step != 0:
            continue
        print(f"  {n + 1}: {f}")
        frame = cv2.imread(f)
        if frame is None:
            raise ValueError(f"Failed to read image file: {f}")
        frames_yielded += 1
        yield frame, n + 1

    if frames_yielded == 0:
        raise ValueError(f"No frames yielded from '{input_path}'. "
                         f"frame_step ({frame_step}) may be too large for {len(files)} file(s).")


def video_frames(video_file, frame_step=1):
    """Yield frames from a video file"""
    vf = cv2.VideoCapture(video_file)
    if not vf.isOpened():
        vf.release()
        raise ValueError(f"Failed to open video file: {video_file}")

    print("opened video:", video_file)
    frame_number = 0
    frames_yielded = 0
    while True:
        ret, frame = vf.read()
        if not ret:
            break
        frame_number += 1
        if (frame_number - 1) % frame_step == 0:
            frames_yielded += 1
            yield frame, frame_number
    vf.release()

    if frames_yielded == 0:
        raise ValueError(f"No frames yielded from '{video_file}'. "
                         f"Video may be empty or frame_step ({frame_step}) is too large.")


def get_image_files(path):
    """Get list of image files from a path (file, directory, or glob pattern)"""
    if os.path.isdir(path):
        # Directory: find all common image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(path, ext)))
            files.extend(glob.glob(os.path.join(path, ext.upper())))
        return sorted(files)
    elif os.path.isfile(path):
        # Single file or video
        return [path]
    else:
        # Treat as glob pattern
        return sorted(glob.glob(path))


def stereo_frames_separate(left_path, right_path, frame_step=1):
    """Yield paired frames from separate left and right image sources"""
    left_files = get_image_files(left_path)
    right_files = get_image_files(right_path)

    if len(left_files) == 0:
        raise ValueError(f"No images found in left path: {left_path}")
    if len(right_files) == 0:
        raise ValueError(f"No images found in right path: {right_path}")

    # Check if inputs are videos
    left_is_video = len(left_files) == 1 and cv2.VideoCapture(left_files[0]).isOpened()
    right_is_video = len(right_files) == 1 and cv2.VideoCapture(right_files[0]).isOpened()

    if left_is_video and right_is_video:
        # Both are videos
        left_cap = cv2.VideoCapture(left_files[0])
        right_cap = cv2.VideoCapture(right_files[0])
        frame_number = 0
        while True:
            ret_l, left_frame = left_cap.read()
            ret_r, right_frame = right_cap.read()
            if not ret_l or not ret_r:
                break
            frame_number += 1
            if (frame_number - 1) % frame_step == 0:
                yield left_frame, right_frame, frame_number
        left_cap.release()
        right_cap.release()
    else:
        # Image lists
        if len(left_files) != len(right_files):
            print(f"Warning: left ({len(left_files)}) and right ({len(right_files)}) "
                  f"image counts differ. Using minimum count.")
        num_frames = min(len(left_files), len(right_files))
        for n in range(0, num_frames, frame_step):
            print(f"Frame {n}: {left_files[n]}, {right_files[n]}")
            left_frame = cv2.imread(left_files[n])
            right_frame = cv2.imread(right_files[n])
            if left_frame is None:
                print(f"Warning: Could not read left image: {left_files[n]}")
                continue
            if right_frame is None:
                print(f"Warning: Could not read right image: {right_files[n]}")
                continue
            yield left_frame, right_frame, n


def detect_grid_stereo_separate(left_path, right_path, grid_size=(6,5),
                                 frame_step=1, gui=False, bayer=False):
    """Detect a grid in each frame from separate left/right image sources"""

    # Dicts to store corner points from all the images.
    left_data = {}
    right_data = {}
    img_shape = None

    print(f"left: {left_path}")
    print(f"right: {right_path}")

    for left_frame, right_frame, frame_number in stereo_frames_separate(
            left_path, right_path, frame_step):

        if bayer:
            left_gray = cv2.cvtColor(left_frame[:,:,0], cv2.COLOR_BayerBG2GRAY)
            right_gray = cv2.cvtColor(right_frame[:,:,0], cv2.COLOR_BayerBG2GRAY)
        else:
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        if img_shape is None:
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
                left_color = cv2.cvtColor(left_frame[:,:,0], cv2.COLOR_BayerBG2BGR)
                right_color = cv2.cvtColor(right_frame[:,:,0], cv2.COLOR_BayerBG2BGR)
            else:
                left_color = left_frame.copy()
                right_color = right_frame.copy()
            if left_corners is not None:
                cv2.drawChessboardCorners(left_color, grid_size, left_corners, True)
            if right_corners is not None:
                cv2.drawChessboardCorners(right_color, grid_size, right_corners, True)
            # Stack images side by side for display
            combined = np.hstack([left_color, right_color])
            cv2.imshow('img', combined)
            cv2.waitKey(-1)

    print("done")
    if gui:
        cv2.destroyAllWindows()

    if img_shape is None:
        raise ValueError(f"No frames were processed from left='{left_path}' and "
                         f"right='{right_path}'. Check that the input paths are correct.")

    return img_shape, left_data, right_data


def detect_grid_video(input_path, grid_size=(6,5), frame_step=1, gui=False, bayer=False):
    """Detect a grid in each frame of video or image(s)"""

    # Dicts to store corner points from all the images.
    left_data = {}
    right_data = {}
    img_shape = None

    # Choose appropriate frame source based on input type
    if os.path.isfile(input_path) and is_video_file(input_path):
        frame_source = video_frames(input_path, frame_step)
    else:
        frame_source = image_frames(input_path, frame_step)

    for frame, frame_number in frame_source:

        left_img = frame[:, 0:frame.shape[1] // 2]
        right_img = frame[:, frame.shape[1] // 2:]
        if bayer:
            left_gray = cv2.cvtColor(left_img[:,:,0], cv2.COLOR_BayerBG2GRAY)
            right_gray = cv2.cvtColor(right_img[:,:,0], cv2.COLOR_BayerBG2GRAY)
        else:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
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

    if img_shape is None:
        raise ValueError(f"No frames were processed from '{input_path}'. "
                         f"Check that the input video or image path is correct.")

    return img_shape, left_data, right_data


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
    K = np.matrix([[1000, 0 , img_shape[1]/2],[0, 1000, img_shape[0]/2],[0, 0, 1]], dtype=np.float32)
    d = np.matrix([0,0,0,0], dtype=np.float32)
    cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
    print("initial calibration error: ", cal_result[0])

    # per frame analysis
    #frames, imgpoints, objpoints = evaluate_error(imgpoints, object_points, frames, cal_result)

    ret, mtx, dist, rvecs, tvecs = cal_result
    aspect_ratio = mtx[0,0] / mtx[1,1]
    print("aspect ratio: ",aspect_ratio)
    if 1.0 - min(aspect_ratio, 1.0/aspect_ratio) < 0.01:
        print("fixing aspect ratio at 1.0")
        flags += cv2.CALIB_FIX_ASPECT_RATIO
        cal_result = cv2.calibrateCamera(objpoints, imgpoints, img_shape, K, d, flags=flags)
        ret, mtx, dist, rvecs, tvecs = cal_result
        print("Fixed aspect ratio error: ", cal_result[0])

    pp = np.array([mtx[0,2], mtx[1,2]])
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
    usage = "usage: %prog [options] [stitched_video_or_images]" + os.linesep + os.linesep
    usage += "  Estimate stereo calibration from calibration target images." + os.linesep
    usage += os.linesep
    usage += "  Input modes:" + os.linesep
    usage += "    1. Stitched stereo (default): Provide a single video file or image" + os.linesep
    usage += "       glob pattern where left and right images are horizontally" + os.linesep
    usage += "       concatenated (left on left half, right on right half)." + os.linesep
    usage += "       Example: %prog calibration_video.mp4" + os.linesep
    usage += "       Example: %prog 'calibration_images/*.png'" + os.linesep
    usage += os.linesep
    usage += "    2. Separate stereo: Use --left and --right options to specify" + os.linesep
    usage += "       separate paths for left and right camera images. Each path" + os.linesep
    usage += "       can be a video file, directory, or glob pattern." + os.linesep
    usage += "       Example: %prog --left left_video.mp4 --right right_video.mp4" + os.linesep
    usage += "       Example: %prog --left ./left_images/ --right ./right_images/" + os.linesep
    usage += "       Example: %prog --left 'left/*.png' --right 'right/*.png'" + os.linesep
    parser = OptionParser(usage=usage)

    parser.add_option("-l", "--left", type='string', default=None,
                      action="store", dest="left_path",
                      help="left camera images (video, directory, or glob pattern)")
    parser.add_option("-r", "--right", type='string', default=None,
                      action="store", dest="right_path",
                      help="right camera images (video, directory, or glob pattern)")
    parser.add_option("-b", "--bayer", default=False,
                      action="store_true", dest="bayer",
                      help="input images are Bayer patterned")
    parser.add_option("-c", "--corners-file", type='string', default=None,
                      action="store", dest="corners_file",
                      help="corner file input path")
    parser.add_option("-i", "--intr-file", type='string', default=None,
                      action="store", dest="intr_file",
                      help="input intrinsics file if only recomputing extr")
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
    parser.add_option("-j", "--json", type='string', default=None,
                      action="store", dest="json_file",
                      help="write as kwiver camera_rig_io compatible json")

    (options, args) = parser.parse_args()

    # Determine input mode
    use_separate_stereo = options.left_path is not None or options.right_path is not None

    if use_separate_stereo:
        if options.left_path is None or options.right_path is None:
            parser.error("Both --left and --right must be specified for separate stereo mode")
        if len(args) > 0:
            parser.error("Cannot specify positional argument with --left/--right options")
    else:
        if len(args) < 1:
            parser.error("Must specify either a stitched video/image path or --left and --right options")

    grid_size = (options.grid_x, options.grid_y)

    img_shape = None
    if options.corners_file:
        if os.path.exists(options.corners_file):
            data = np.load(options.corners_file, allow_pickle=True)
            img_shape = tuple(data["img_shape"])
            left_data = data["left_data"].item()
            right_data = data["right_data"].item()

    if img_shape == None:
        if use_separate_stereo:
            img_shape, left_data, right_data = detect_grid_stereo_separate(
                options.left_path, options.right_path, grid_size,
                options.frame_step, options.gui, options.bayer)
        else:
            video_file = args[0]
            img_shape, left_data, right_data = detect_grid_video(
                video_file, grid_size, options.frame_step,
                options.gui, options.bayer)
        if options.corners_file:
            np.savez(options.corners_file, img_shape=img_shape,
                     left_data=left_data, right_data=right_data)

    print("computing calibration")
    objp = make_object_points(grid_size) * options.square_size
    (_, K_left, dist_left, _, _), _ = calibrate_single_camera(left_data, objp, img_shape)
    (_, K_right, dist_right, _, _), _ = calibrate_single_camera(right_data, objp, img_shape)

    # write the intrinsics file
    if not options.intr_file:
        fs = cv2.FileStorage("intrinsics.yml", cv2.FILE_STORAGE_WRITE)
        if (fs.isOpened()):
            fs.write("M1", K_left)
            fs.write("D1", dist_left)
            fs.write("M2", K_right)
            fs.write("D2", dist_right)
        fs.release()
    else:
        npz_dict = dict(np.load(options.intr_file))
        K_left = npz_dict[ 'cameraMatrixL' ]
        K_right = npz_dict[ 'cameraMatrixR' ]
        dist_left = npz_dict[ 'distCoeffsL' ]
        dist_right = npz_dict[ 'distCoeffsR' ]

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

    # write the extrinsics file
    fs = cv2.FileStorage("extrinsics.yml", cv2.FILE_STORAGE_WRITE)
    if (fs.isOpened()):
        fs.write("R", R)
        fs.write("T", T)
        fs.write("R1", R1)
        fs.write("R2", R2)
        fs.write("P1", P1)
        fs.write("P2", P2)
        fs.write("Q", Q)
    fs.release()

    # write joint npz file usable with default VIAME measurement pipeline
    npz_dict = dict()
    npz_dict[ 'cameraMatrixL' ] = K_left
    npz_dict[ 'cameraMatrixR' ] = K_right
    npz_dict[ 'distCoeffsL' ] = dist_left
    npz_dict[ 'distCoeffsR' ] = dist_right
    npz_dict[ 'R' ] = R
    npz_dict[ 'T' ] = T
    np.savez("calibration.npz", **npz_dict)

    # if requested, write as KWIVER camera_rig_io-compatible json file
    if options.json_file:
        json_dict = dict()
        json_dict['T'] = T.flatten().tolist()
        json_dict['R'] = R.flatten().tolist()
        for (m, d, side) in ([K_left, dist_left, 'left'], [K_right, dist_right, 'right']):

            json_dict[f'fx_{side}'] = m[0][0]
            json_dict[f'fy_{side}'] = m[1][1]
            json_dict[f'cx_{side}'] = m[0][2]
            json_dict[f'cy_{side}'] = m[1][2]

            json_dict[f'k1_{side}'] = d[0][0]
            json_dict[f'k2_{side}'] = d[0][1]
            json_dict[f'p1_{side}'] = d[0][2]
            json_dict[f'p2_{side}'] = d[0][3]

        with open(options.json_file, 'w') as fh:
            fh.write(json.dumps(json_dict))

if __name__ == "__main__":
    main()
