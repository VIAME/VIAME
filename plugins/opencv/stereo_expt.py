#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# -*- coding: utf-8 -*-
"""
Experimental scripts
"""

import cv2
import numpy as np
import ubelt as ub
import sklearn.metrics
import scipy.io
import pandas as pd

from os.path import expanduser
from . import stereo_algos as ctalgo

def to_mat_format():
    import pandas as pd
    measure_fpath = 'measurements_haul83.csv'
    py_df = pd.DataFrame.from_csv(measure_fpath, index_col=None)
    py_df['fishlen'] = py_df['fishlen'] / 10
    bbox_pts1 = py_df['box_pts1'].map(lambda p: eval(p.replace(';', ','), np.__dict__))
    bbox_pts2 = py_df['box_pts2'].map(lambda p: eval(p.replace(';', ','), np.__dict__))

    bbox_pts1 = np.array(bbox_pts1.values.tolist())
    bbox_pts2 = np.array(bbox_pts2.values.tolist())

    X = bbox_pts1.T[0].T
    Y = bbox_pts1.T[1].T
    X = pd.DataFrame(X, columns=['LX1', 'LX2', 'LX3', 'LX4'])
    Y = pd.DataFrame(Y, columns=['LY1', 'LY2', 'LY3', 'LY4'])
    py_df.join(X.join(Y))

    X = bbox_pts2.T[0].T
    Y = bbox_pts2.T[1].T
    X = pd.DataFrame(X, columns=['RX1', 'RX2', 'RX3', 'RX4'])
    Y = pd.DataFrame(Y, columns=['RY1', 'RY2', 'RY3', 'RY4'])
    py_df = py_df.join(X.join(Y))

    py_df = py_df.rename(columns={
        'error': 'Err',
        'fishlen': 'fishLength',
        'range': 'fishRange',
    })
    py_df.drop(['box_pts1', 'box_pts2'], axis=1, inplace=True)
    py_df.to_csv('haul83_py_results.csv')
    pass


def compare_results():
    print('Comparing results')
    import pandas as pd
    from tabulate import tabulate

    # Read in output of demo script
    measure_fpath = 'measurements_haul83.csv'
    py_df = pd.DataFrame.from_csv(measure_fpath, index_col=None)
    # Convert python length output from mm into cm for consistency
    py_df['fishlen'] = py_df['fishlen'] / 10
    py_df['current_frame'] = py_df['current_frame'].astype(np.int64)

    # janky CSV parsing
    py_df['box_pts1'] = py_df['box_pts1'].map(lambda p: eval(p.replace(';', ','), np.__dict__))
    py_df['box_pts2'] = py_df['box_pts2'].map(lambda p: eval(p.replace(';', ','), np.__dict__))

    py_df['obox1'] = [ctalgo.OrientedBBox(*cv2.minAreaRect(pts[:, None, :].astype(np.int64)))
                      for pts in py_df['box_pts1']]
    py_df['obox2'] = [ctalgo.OrientedBBox(*cv2.minAreaRect(pts[:, None, :].astype(np.int64)))
                      for pts in py_df['box_pts2']]
    py_df.drop(['box_pts1', 'box_pts2'], axis=1, inplace=True)

    # Remap to matlab names
    py_df = py_df.rename(columns={
        'error': 'Err',
        'fishlen': 'fishLength',
        'range': 'fishRange',
    })

    # Load matlab results
    mat_df = _read_kresimir_results()

    FORCE_COMPARABLE_RANGE = True
    # FORCE_COMPARABLE_RANGE = False
    if FORCE_COMPARABLE_RANGE:
        # Be absolutely certain we are in comparable regions (may slightly bias
        # results, against python and in favor of matlab)
        min_frame = max(mat_df.current_frame.min(), py_df.current_frame.min())
        max_frame = min(mat_df.current_frame.max(), py_df.current_frame.max())
        print('min_frame = {!r}'.format(min_frame))
        print('max_frame = {!r}'.format(max_frame))

        mat_df = mat_df[(mat_df.current_frame >= min_frame) &
                        (mat_df.current_frame <= max_frame)]
        py_df = py_df[(py_df.current_frame >= min_frame) &
                      (py_df.current_frame <= max_frame)]

    intersect_frames = np.int64ersect1d(mat_df.current_frame, py_df.current_frame)
    print('intersecting frames = {} / {} (matlab)'.format(
        len(intersect_frames), len(set(mat_df.current_frame))))
    print('intersecting frames = {} / {} (python)'.format(
        len(intersect_frames), len(set(py_df.current_frame))))

    #  Reuse the hungarian algorithm implementation from ctalgo
    min_assign = ctalgo.StereoLengthMeasurments.minimum_weight_assignment

    correspond = []
    for f in intersect_frames:
        pidxs = np.where(py_df.current_frame == f)[0]
        midxs = np.where(mat_df.current_frame == f)[0]

        pdf = py_df.iloc[pidxs]
        mdf = mat_df.iloc[midxs]

        ppts1 = np.array([o.center for o in pdf['obox1']])
        mpts1 = np.array([o.center for o in mdf['obox1']])

        ppts2 = np.array([o.center for o in pdf['obox2']])
        mpts2 = np.array([o.center for o in mdf['obox2']])

        dists1 = sklearn.metrics.pairwise.pairwise_distances(ppts1, mpts1)
        dists2 = sklearn.metrics.pairwise.pairwise_distances(ppts2, mpts2)

        # arbitrarilly chosen threshold
        thresh = 100
        for i, j in min_assign(dists1):
            d1 = dists1[i, j]
            d2 = dists2[i, j]
            if d1 < thresh and d2 < thresh and abs(d1 - d2) < thresh / 4:
                correspond.append((pidxs[i], midxs[j]))
    correspond = np.array(correspond)

    # pflags = np.array(ub.boolmask(correspond.T[0], len(py_df)))
    mflags = np.array(ub.boolmask(correspond.T[1], len(mat_df)))
    # print('there are {} detections that seem to be in common'.format(len(correspond)))
    # print('The QC flags of the common detections are:       {}'.format(
    #     ub.dict_hist(mat_df[mflags]['QC'].values)))
    # print('The QC flags of the other matlab detections are: {}'.format(
    #     ub.dict_hist(mat_df[~mflags]['QC'].values)))

    print('\n\n----\n## All stats\n')
    print(ub.codeblock(
        '''
        Overall, the matlab script made {nmat} length measurements and the
        python script made {npy} length measurements.  Here is a table
        summarizing the average lengths / ranges / errors of each script:
        ''').format(npy=len(py_df), nmat=len(mat_df)))
    stats = pd.DataFrame(columns=['python', 'matlab'])
    for key in ['fishLength', 'fishRange', 'Err']:
        stats.loc[key, 'python'] = '{:6.2f} ± {:6.2f}'.format(py_df[key].mean(), py_df[key].std())
        stats.loc[key, 'matlab'] = '{:6.2f} ± {:6.2f}'.format(mat_df[key].mean(), mat_df[key].std())

    stats.loc['nTotal', 'python'] = '{}'.format(len(py_df))
    stats.loc['nTotal', 'matlab'] = '{}'.format(len(mat_df))
    print(tabulate(stats, headers='keys', tablefmt='psql', stralign='right'))

    print('\n\n----\n## Only COMMON detections\n')
    py_df_c = py_df.iloc[correspond.T[0]]
    mat_df_c = mat_df.iloc[correspond.T[1]]
    stats = pd.DataFrame(columns=['python', 'matlab'])
    for key in ['fishLength', 'fishRange', 'Err']:
        stats.loc[key, 'python'] = '{:6.2f} ± {:6.2f}'.format(py_df_c[key].mean(), py_df_c[key].std())
        stats.loc[key, 'matlab'] = '{:6.2f} ± {:6.2f}'.format(mat_df_c[key].mean(), mat_df_c[key].std())

    stats.loc['nTotal', 'python'] = '{}'.format(len(py_df_c))
    stats.loc['nTotal', 'matlab'] = '{}'.format(len(mat_df_c))

    print(ub.codeblock(
        '''
        Now, we investigate how many dections matlab and python made in common.
        (Note, choosing which dections in one version correspond to which in
         another is done using a heuristic based on distances between bbox
         centers and a thresholded minimum assignment problem).

        Python made {npy_c}/{nmat} = {percent:.2f}% of the detections matlab made

        ''').format(npy_c=len(py_df_c), nmat=len(mat_df),
                    percent=100 * len(py_df_c) / len(mat_df)))
    print(tabulate(stats, headers='keys', tablefmt='psql', stralign='right'))

    print('\n\n----\n## Evaulation using the QC code\n')
    hist_hit = ub.dict_hist(mat_df[mflags]['QC'].values)
    hist_miss = ub.dict_hist(mat_df[~mflags]['QC'].values)
    print(ub.codeblock(
        '''
        However, not all of those matlab detections were good. Because we have
        detections in corrsepondences with each other we can assign the python
        detections QC codes.

        Here is a histogram of the QC codes for these python detections:
        {}
        (Note: read histogram as <QC-code>: <frequency>)

        Here is a histogram of the other matlab detections that python did not
        find:
        {}

        To summarize:
            python correctly rejected {:.2f}% of the matlab QC=0 detections
            python correctly accepted {:.2f}% of the matlab QC=1 detections
            python correctly accepted {:.2f}% of the matlab QC=2 detections

            Note, that because python made detections that matlab did not make,
            the remaining {} detections may be right or wrong, but there is
            no way to tell from this analysis.

        Lastly, here are the statistics for the common detections that had a
        non-zero QC code.
        ''').format(
            ub.repr2(hist_hit, nl=1),
            ub.repr2(hist_miss, nl=1),
            100 * hist_miss[0] / (hist_hit[0] + hist_miss[0]),
            100 * hist_hit[1] / (hist_hit[1] + hist_miss[1]),
            100 * hist_hit[2] / (hist_hit[2] + hist_miss[2]),
            len(py_df) - len(py_df_c)
                   )
    )

    is_qc = (mat_df_c['QC'] > 0).values
    mat_df_c = mat_df_c[is_qc]
    py_df_c = py_df_c[is_qc]
    stats = pd.DataFrame(columns=['python', 'matlab'])
    for key in ['fishLength', 'fishRange', 'Err']:
        stats.loc[key, 'python'] = '{:6.2f} ± {:6.2f}'.format(py_df_c[key].mean(), py_df_c[key].std())
        stats.loc[key, 'matlab'] = '{:6.2f} ± {:6.2f}'.format(mat_df_c[key].mean(), mat_df_c[key].std())

    stats.loc['nTotal', 'python'] = '{}'.format(len(py_df_c))
    stats.loc['nTotal', 'matlab'] = '{}'.format(len(mat_df_c))
    print(tabulate(stats, headers='keys', tablefmt='psql', stralign='right'))


def _read_kresimir_results():
    # Load downloaded matlab csv results
    mat = scipy.io.loadmat(expanduser('~/data/opencv_stereo_sample_data/Haul_83/Haul_083_qcresult.mat'))
    header = ub.readfrom(expanduser('~/data/opencv_stereo_sample_data/Haul_83/mat_file_header.csv')).strip().split(',')
    data = mat['lengthsqc']

    mat_df = pd.DataFrame(data, columns=header)
    mat_df['current_frame'] = mat_df['current_frame'].astype(np.int64)
    mat_df['Species'] = mat_df['Species'].astype(np.int64)
    mat_df['QC'] = mat_df['QC'].astype(np.int64)

    # Transform so each row corresponds to one set of (x, y) points per detection
    bbox_cols1 = ['LX1', 'LX2', 'LX3', 'LX4', 'LY1', 'LY2', 'LY3', 'LY4', 'Lar', 'LboxL', 'WboxL', 'aveL']
    bbox_pts1 = mat_df[bbox_cols1[0:8]]  # NOQA
    bbox_pts1_ = bbox_pts1.values
    bbox_pts1_ = bbox_pts1_.reshape(len(bbox_pts1_), 2, 4).transpose((0, 2, 1))

    bbox_cols2 = ['RX1', 'RX2', 'RX3', 'RX4', 'RY1', 'RY2', 'RY3', 'RY4', 'Rar', 'LboxR', 'WboxR', 'aveW']
    bbox_pts2 = mat_df[bbox_cols2]  # NOQA
    bbox_pts2 = mat_df[bbox_cols2[0:8]]  # NOQA
    bbox_pts2_ = bbox_pts2.values
    bbox_pts2_ = bbox_pts2_.reshape(len(bbox_pts2_), 2, 4).transpose((0, 2, 1))

    # Convert matlab bboxes into python-style bboxes
    mat_df['obox1'] = [ctalgo.OrientedBBox(*cv2.minAreaRect(pts[:, None, :].astype(np.int64)))
                       for pts in bbox_pts1_]
    mat_df['obox2'] = [ctalgo.OrientedBBox(*cv2.minAreaRect(pts[:, None, :].astype(np.int64)))
                       for pts in bbox_pts2_]

    mat_df.drop(bbox_cols2, axis=1, inplace=True)
    mat_df.drop(bbox_cols1, axis=1, inplace=True)
    return mat_df


if __name__ == '__main__':
    r"""
    CommandLine:
        python ~/code/VIAME/plugins/opencv/python/opencv_expt.py
    """
    compare_results()
