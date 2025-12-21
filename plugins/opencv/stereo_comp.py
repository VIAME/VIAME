
''' attempt at writing a triangulation function to mimic matlab camera calibration toolbox'''

from scipy.io import matlab

import numpy as np
import string
import cv2

class pyStereoComp(object):

    """

    import sys
    sys.path.append('/home/joncrall/code/VIAME/plugins/opencv/python')
    from pyStereoComp import *
    import glob
    from os.path import expanduser, join
    data_fpath = expanduser('~/data/autoprocess_test_set')
    fname = join(data_fpath, 'cal_201608.mat')

    self = pyStereoComp()
    self.importCalData(cal_fpath)
    """

    def __init__(self):
        self.calData = {}
        self.mode = 'matlab'  # teh default

    def importCalData(self, fname):
        if self.mode == 'matlab':
            try:
                self.calData = matlab.loadmat(fname)
                #--- Rotation matrix corresponding to the rigid motion between left and right cameras:
                om = self.calData['om']
                R = self.rodrigues(om)
                self.calData.update({'R': R})

                return True,  self.calData
            except Exception as e:
                try:
                    import scipy.io
                    cal_data = scipy.io.loadmat(fname)
                    cal = cal_data['Cal']

                    (om, T, fc_left, fc_right, cc_left, cc_right, kc_left, kc_right,
                     alpha_c_left, alpha_c_right) = cal[0][0]

                    self.calData = {
                        'om': om,
                        'T': T,
                        'fc_left': fc_left,
                        'fc_right': fc_right,
                        'cc_left': cc_left,
                        'cc_right': cc_right,
                        'kc_left': kc_left,
                        'kc_right': kc_right,
                        'alpha_c_left': alpha_c_left,
                        'alpha_c_right': alpha_c_right,
                    }

                    #--- Rotation matrix corresponding to the rigid motion between left and right cameras:
                    om = self.calData['om']
                    R = self.rodrigues(om)
                    self.calData.update({'R': R})
                    return True,  self.calData
                except Exception as ex:
                    msg = ''.join(s for s in str(e) if s in string.printable)
                    return False,  msg

            fc = self.calData['fc_left'].ravel()
            cc = self.calData['cc_left'].ravel()
            alpha_c = self.calData['alpha_c_left'].ravel()
            KL = np.array([
                [fc[0], alpha_c * fc[0], cc[0]],
                [    0,           fc[1], cc[1]],
                [    0,               0,     1],
            ])

            fc = self.calData['fc_right'].ravel()
            cc = self.calData['cc_right'].ravel()
            alpha_c = self.calData['alpha_c_right'].ravel()
            KR = np.array([
                [fc[0], alpha_c * fc[0], cc[0]],
                [    0,           fc[1], cc[1]],
                [    0,               0,     1],
            ])
            self.calData['cameraMatrixL'] = KL
            self.calData['cameraMatrixR'] = KR

            self.calData['distCoeffsL'] = self.calData['kc_left'].ravel()
            self.calData['distCoeffsR'] = self.calData['kc_right'].ravel()

        elif self.mode == 'openCV':
            try:
                npzfileData = np.load(fname)
                self.calData = {
                    'cameraMatrixL': npzfileData['cameraMatrixL'],
                    'distCoeffsL': npzfileData['distCoeffsL'],
                    'kc_left': npzfileData['distCoeffsL'][0],
                    'kc_right': npzfileData['distCoeffsR'][0],
                    'cameraMatrixR': npzfileData['cameraMatrixR'],
                    'distCoeffsR': npzfileData['distCoeffsR'],
                    'R': npzfileData['R'],
                    'T': npzfileData['T']
                }
                if 'F' in npzfileData:
                    self.calData.update({'F': npzfileData['F']})
                R = self.calData['R']
                #om = self.rodrigues(R)
                # self.calData.update({'om':om})
                calmat = npzfileData['cameraMatrixL']
                self.calData.update(
                    {'fc_left': np.array([[calmat[0, 0]], [calmat[1, 1]]])})
                self.calData.update(
                    {'cc_left': np.array([[calmat[0, 2]], [calmat[1, 2]]])})
                self.calData.update({'alpha_c_left': np.array([[calmat[0, 1]]])})

                calmat = npzfileData['cameraMatrixR']
                self.calData.update(
                    {'fc_right': np.array([[calmat[0, 0]], [calmat[1, 1]]])})
                self.calData.update(
                    {'cc_right': np.array([[calmat[0, 2]], [calmat[1, 2]]])})
                self.calData.update({'alpha_c_right': np.array([[calmat[0, 1]]])})

                return True,  self.calData
            except Exception as e:
                msg = ''.join(s for s in str(e) if s in string.printable)
                return False,  msg
        else:
            return False,  'mode unrecognized'

    def triangulatePoint(self, xL, xR, mode=None):
        """
        Args:
            xL (ndarray): [2 x N] array of N 2D points in left cam pixels
            xR (ndarray): [2 x N] array of N 2D points in right cam pixels
        """
        if mode is None:
            mode = self.mode

        if mode.lower() == 'opencv':
            if False:
                hxL = np.zeros((xL.shape[1], 1, 2))
                hxR = np.zeros((xL.shape[1], 1, 2))
                for i in range(xL.shape[1]):
                    hxL[i, 0, 0] = xL[0, i]
                    hxL[i, 0, 1] = xL[1, i]
                    hxR[i, 0, 0] = xR[0, i]
                    hxR[i, 0, 1] = xR[1, i]
            else:
                # Much easier way to do above code
                hxL = xL.T[:, None, :]
                hxR = xR.T[:, None, :]

            KL = self.calData['cameraMatrixL']
            KR = self.calData['cameraMatrixR']
            kcR = self.calData['distCoeffsR']
            kcL = self.calData['distCoeffsL']

            R, T = self.calData['R'], self.calData['T']

            # THIS SEEMS WRONG
            PL = np.hstack((np.eye(3), np.zeros((3, 1))))
            PR = np.hstack((R, T))
            # PL = KL.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
            # PR = KR.dot(np.hstack((R, T)))

            imgptsL = cv2.undistortPoints(hxL, KL, kcL)
            imgptsR = cv2.undistortPoints(hxR, KR, kcR)

            import utool as ut
            print('imgptsL =\n{}'.format(ut.repr2(imgptsL[:, 0, :].T, precision=3)))
            print('imgptsR =\n{}'.format(ut.repr2(imgptsR[:, 0, :].T, precision=3)))

            XL = cv2.triangulatePoints(PL, PR, imgptsL, imgptsR)
            XL /= XL[3]
            XL = XL[0:3]
            XR = np.dot(R, XL) + T

        elif mode == 'matlab':
            '''% [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right),
                %
                % Function that computes the position of a set on N points given the left and right image projections.
                % The cameras are assumed to be calibrated, intrinsically, and extrinsically.
                %
                % Input:
                %           xL: 2xN matrix of pixel coordinates in the left image
                %           xR: 2xN matrix of pixel coordinates in the right image
                %           om,T: rotation vector and translation vector between right and left cameras (output of stereo calibration)
                %           fc_left,cc_left,...: intrinsic parameters of the left camera  (output of stereo calibration)
                %           fc_right,cc_right,...: intrinsic parameters of the right camera (output of stereo calibration)
                %
                % Output:
                %
                %           XL: 3xN matrix of coordinates of the points in the left camera reference frame
                %           XR: 3xN matrix of coordinates of the points in the right camera reference frame
                %
                % Note: XR and XL are related to each other through the rigid motion equation: XR = R * XL + T, where R = rodrigues(om)
                % For more information, visit http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html
                %
                %
                % (c) Jean-Yves Bouguet - Intel Corporation - April 9th, 2003
                '''
            #--- Normalize the image projection according to the intrinsic parameters of the left and right cameras
            xt = self.normalizePixel(
                xL, self.calData['fc_left'], self.calData['cc_left'], self.calData['kc_left'], self.calData['alpha_c_left'])
            xtt = self.normalizePixel(
                xR, self.calData['fc_right'], self.calData['cc_right'], self.calData['kc_right'], self.calData['alpha_c_right'])

            import utool as ut
            print('xt =\n{}'.format(ut.repr2(xt, precision=3)))
            print('xtt =\n{}'.format(ut.repr2(xtt, precision=3)))

            T = self.calData['T']
            R = self.calData['R']
            #R=np.array([[0.982261529882744,       -0.0231885098133093,         0.186076811895112],[0.0236366236373557,         0.999720597968969,      -0.00018978828345735],[-0.186020420748467,       0.00458464930006644,          0.98253518209546]])
            #--- Extend the normalized projections in homogeneous coordinates
            g = np.array(np.ones((1, xt.shape[1])))
            xt = np.vstack((xt, g))
            xtt = np.vstack((xtt, g))

            #--- Number of points:
            N = xt.shape[1]

            #--- Triangulation of the rays in 3D space:

            u = np.dot(R,  xt)

            n_xt2 = (xt * xt).sum(axis=0)
            n_xtt2 = (xtt * xtt).sum(axis=0)

            T_vect = np.tile(T, (1, N))

            DD = n_xt2 * n_xtt2 - (u * xtt).sum(axis=0)**2

            dot_uT = (u * T_vect).sum(axis=0)
            dot_xttT = (xtt * T_vect).sum(axis=0)
            dot_xttu = (u * xtt).sum(axis=0)

            NN1 = dot_xttu * dot_xttT - n_xtt2 * dot_uT
            NN2 = n_xt2 * dot_xttT - dot_uT * dot_xttu

            Zt = NN1 / DD
            Ztt = NN2 / DD

            X1 = xt * np.tile(Zt, (3,  1))
            X2 = np.dot(R.T, (xtt * np.tile(Ztt, (3, 1)) - T_vect))

            #--- Left coordinates:
            XL = 0.5 * (X1 + X2)

            #--- Right coordinates:
            XR = np.dot(R, XL) + T_vect

        import utool as ut
        print('XL =\n{}'.format(ut.repr2(XL, precision=3)))
        print('XR =\n{}'.format(ut.repr2(XR, precision=3)))
        print('----')
        return (XL, XR)

    def normalizePixel(self, x_kk, fc, cc, kc, alpha_c):

        x_distort = np.array([(x_kk[0, :] - cc[0]) / fc[0], (x_kk[1, :] - cc[1]) / fc[1]])
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

    def rodrigues(self, om):
        """
        This works exactly the same as `cv2.Rodrigues(om)[0]`
        """

        (m, n) = om.shape
        eps = 2.22044604925031e-016
        bigeps = 10e+20 * eps
        if (m == 1 and n == 3) or (m == 3 and n == 1):  # it is a rotation vector
            theta = np.linalg.norm(om)
            if theta < eps:
                R = np.eye(3)

                # if nargout > 1,

                dRdin = np.array([[0, 0, 0], [0, 0, 1], [
                              0, -1, 0], [0, 0, -1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 0]])
                dRdin = dRdin.T
            else:
                if n == len(om):
                    om = om.T  # make it a column vec. if necess.

                dm3din = np.vstack((np.eye(3), om.T / theta))
                omega = om / theta
                dm2dm3 = np.vstack(
                    (np.hstack((np.eye(3) / theta,  -om / theta**2)), np.array([[0, 0, 0, 1]])))
                alpha = np.cos(theta)
                beta = np.sin(theta)
                gamma = 1 - np.cos(theta)
                omegav = np.array([[0,  -omega[2],  omega[1]], [omega[2], 0,  -omega[0]], [-omega[1],  omega[0],  0]])
                A = np.dot(omega, omega.T)

                pt2 = np.zeros((3, 4))
                pt2[0, 3] = -np.sin(theta)
                pt2[1, 3] = np.cos(theta)
                pt2[2, 3] = np.sin(theta)
                z = np.array([[0,  0,   0,   0,  0,  1,  0,  -1,  0],
                              [0,  0,  -1,   0,  0,  0,  1,   0,  0],
                              [0,  1,   0,  -1,  0,  0,  0,   0,  0],
                              [0,  0,   0,   0,  0,  0,  0,   0,  0]])

                w1 = omega[0, 0]
                w2 = omega[1, 0]
                w3 = omega[2, 0]

                pt3 = np.array([[2 * w1, w2, w3, w2, 0, 0, w3, 0, 0],
                                [0, w1, 0, w1, 2 * w2, w3, 0, w3, 0],
                                [0, 0, w1, 0, 0, w2, w1, w2, 2 * w3]])

                dm1dm2 = np.vstack((pt2, z.T, np.hstack((pt3.T, np.zeros((9, 1))))))

                R = np.eye(3) * alpha + omegav * beta + A * gamma

                pt1 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
                pt2 = omegav.flatten(1)
                dRdm1 = np.vstack((pt1, pt2))
                pt3 = beta * np.eye(9)
                pt4 = A.flatten(1)
                dRdm1 = np.vstack((dRdm1, pt4, pt3.T))

                pt5 = gamma * np.eye(9)
                dRdm1 = np.vstack((dRdm1, pt5.T))

                dRdin = np.dot(np.dot(np.dot(dRdm1.T, dm1dm2), dm2dm3), dm3din)

                out = R
                # dout = dRdin

        elif (m == n and m == 3) and (all(((np.linalg.norm(om.T) * om - np.eye(3)) < bigeps)) and (abs(np.linalg.det(om) - 1) < bigeps)):
            R = om

            # project the rotation matrix to SO(3);
            (U, s, V) = np.linalg.svd(R)
            V = V * np.array([[1, 1, 1], [1, 1, 1], [-1, -1, -1]])
            V = V.T
            # R = np.dot(NU, NV.T)
            # z = np.dot(U, V[0,:])

            # R=np.array([[z[0],z[1],z[2]],[z[1],z[2],z[0]],[z[2],z[0],z[1]]]).T

            tr = (np.trace(R) - 1) / 2
            # dtrdR = np.array([[1., 0., 0., 0., 1., 0., 0., 0., 1.]]) / 2
            p = np.acos(tr)
            theta = p.real

            if np.sin(theta) >= 1e-4:

                # dthetadtr = -1 / np.sqrt(1 - tr**2)

                # dthetadR = dthetadtr * dtrdR
                # vth = 1 / (2 * np.sin(theta))
                # dvthdtheta = -vth * np.cos(theta) / np.sin(theta)
                # dvar1dtheta = np.array([[dvthdtheta, 1]]).T

                # dvar1dR = np.dot(dvar1dtheta, dthetadR)

                # om1 = np.array(
                #     [[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]]).T

                # dom1dR = np.array([[0., 0., 0., 0., 0., 1., 0., -1., 0.],
                #                    [0., 0., -1., 0., 0., 0., 1., 0., 0.],
                #                    [0., 1., 0., -1., 0., 0., 0., 0., 0.]]).T
                # dvardR = np.vstack((dom1dR.T, dvar1dR))

                # om = vth * om1

                # domdvar = np.hstack((vth * np.eye(3), om1, np.zeros((3, 1))))
                # dthetadvar = np.array([[0., 0., 0., 0., 1.]])
                # dvar2dvar = np.vstack((domdvar, dthetadvar))

                out = om * theta
                # domegadvar2 = np.hstack((theta * np.eye(3), om))

                # dout = np.dot(np.dot(domegadvar2, dvar2dvar),  dvardR)

            else:

                if tr > 0:          # case norm(om)=0;

                    out = np.array([[0., 0., 0.]]).T

                    # dout = np.array([[0., 0., 0., 0., 0., .5, 0., -.5, 0.],
                    #                  [0., 0., -.5, 0., 0., 0., .5, 0., 0.],
                    #                  [0., .5, 0., -.5, 0., 0., 0., 0., 0.]]).T
                else:
                    # case norm(om)=pi;
                    # Solution by Mike Burl on Feb 27, 2007
                    # This is a better way to determine the signs of the
                    # entries of the rotation vector using a hash table on all
                    # the combinations of signs of a pairs of products (in the
                    # rotation matrix)

                    # Define hashvec and Smat
                    hashvec = np.array(
                        [[0., -1., -3., -9., 9., 3., 1., 13., 5., -7., -11.]]).T
                    Smat = np.array([[1, 1, 1], [1, 0, -1], [0, 1, -1], [1, -1, 0], [1, 1, 0], [
                                 0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, -1], [1, -1, -1], [1, -1, 1]])

                    M = (R + np.eye(3, 3)) / 2
                    uabs = np.sqrt(M[0, 0])
                    vabs = np.sqrt(M[1, 1])
                    wabs = np.sqrt(M[2, 2])

                    mvec = np.array([[M[0, 1], M[1, 2], M[0, 2]]])
                    # robust sign() function
                    s = ((mvec > 1e-4) - (mvec < -1e-4))
                    syn = np.array([[int(s[0, 0]), int(s[0, 1]), int(s[0, 2])]])
                    hash = np.dot(syn, np.array([[9.,  3.,  1.]]).T)
                    hash = hash[0, 0]
                    idx = (hash == hashvec).nonzero()
                    svec = np.array([Smat[idx[0][0], :]]).T

                    out = theta * np.array([[uabs, vabs, wabs]]).T * svec

        else:
            raise Exception(
                'Neither a rotation matrix nor a rotation vector were provided')

        return out

    def computeEpipole(self,  clickLoc,  I,  d=[-100, 100], n=20):
        pt = np.array([[clickLoc.x()], [clickLoc.y()]])
        if self.mode == 'openCV':
            if I == 'L':
                lineParams = cv2.computeCorrespondEpilines(
                    pt, 1, self.calData['F'])
            elif I == 'R':
                lineParams = cv2.computeCorrespondEpilines(
                    pt, 2, self.calData['F'])
            print(lineParams)
        else:
            pass
        h = self.normalizePixel(
            pt, self.calData['fc_left'], self.calData['cc_left'], self.calData['kc_left'], self.calData['alpha_c_left'])
        uo = np.hstack((h[0], h[1], 1))
        T = self.calData['T']
        R = self.calData['R']
        S = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])

        l_epipole = np.dot(np.dot(S, R), uo.transpose())

        KK_right = np.array([
            [self.calData['fc_right'][0][0],
             self.calData['alpha_c_right'][0][0] * self.calData['fc_right'][0][0],
             self.calData['cc_right'][0][0]],
            [0,
             self.calData['fc_right'][1][0],
             self.calData['cc_right'][1][0]],
            [0,  0,  1]])

        if np.sqrt(np.dot(l_epipole[1], l_epipole[1])) > np.sqrt(np.dot(l_epipole[0], l_epipole[0])):
            limit_x_pos = (
                (pt[1][0] + d[1]) - self.calData['cc_right'][0]) / self.calData['fc_right'][0]
            limit_x_neg = (
                (pt[1][0] - d[0]) - self.calData['cc_right'][0]) / self.calData['fc_right'][0]

            x_list = (limit_x_pos - limit_x_neg) * (np.array(range(n)).astype('float') / (float(n) - 1)) + limit_x_neg

            pt = np.cross(np.tile(np.array([l_epipole]).T, (1, n)).T, np.array(
                [np.ones((1, n))[0], np.zeros((1, n))[0], -x_list]).T).T

        else:

            limit_y_pos = (
                (pt[1][0] + d[1]) - self.calData['cc_right'][1]) / self.calData['fc_right'][1]
            limit_y_neg = (
                (pt[1][0] - d[0]) - self.calData['cc_right'][1]) / self.calData['fc_right'][1]

            y_list = (limit_y_pos - limit_y_neg) * \
                (range(n) / (n - 1)) + limit_y_neg

            pt = np.cross(np.tile(l_epipole, (1, n)), np.array(
                [[np.zeros((1, n))], [np.ones(1, n)], [-y_list]]))

        pt = np.vstack((pt[0, :] / pt[2, :], pt[1, :] / pt[2, :]))
        ptd = self.applyDistortion(pt, self.calData['kc_right'])
        epipole = np.dot(KK_right,  np.vstack((ptd, np.ones((1, n)))))

        return epipole[0:2, :]

    def applyDistortion(self, x, k):
        # Complete the distortion vector if you are using the simple distortion model:
        length_k = len(k)
        if length_k < 5:
            k = np.hstack(k, np.zeros((5 - length_k, 1)))
        # d = x.shape
        # if len(d) < 2:
        #     n = 1
        # else:
        #     n = d[1]
        #     pass

        # Add distortion:
        r2 = x[0, :]**2 + x[1, :] ** 2
        r4 = r2**2
        r6 = r2**3
        # Radial distortion:
        cdist = np.array([1 + k[0] * r2 + k[1] * r4 + k[4] * r6])
        xd1 = x * np.dot(np.ones((2, 1)), cdist)
        # tangential distortion:
        a1 = 2 * x[0, :] * x[1, :]
        a2 = r2 + 2 * x[0, :] ** 2
        a3 = r2 + 2 * x[1, :] ** 2
        delta_x = np.vstack((k[2] * a1 + k[3] * a2, k[2] * a3 + k[3] * a1))
        xd = xd1 + delta_x
        return xd

    def projectPoint(self, X, I):
        if self.mode == 'openCV':
            if I == 'L':
                xp, J = cv2.projectPoints(X.T,  np.eye(3), np.zeros(
                    (3, 1)), self.calData['cameraMatrixL'], self.calData['distCoeffsL'])
            elif I == 'R':
                xp, J = cv2.projectPoints(X.T,  np.eye(3), np.zeros(
                    (3, 1)), self.calData['cameraMatrixR'], self.calData['distCoeffsR'])
            if xp.shape[0] < 2:
                xp = xp[0].T
            else:
                xp = np.reshape(xp, (2, 2)).T
        elif self.mode == 'matlab':

            # rigid motion: use 3x3 identity matrix for R and 3x1 0 vector for T
            R = np.eye(3)
            T = np.array([[0], [0], [0]])
            # R = self.calData('R')
            # T = self.calData('T')
            P = np.dot(R, X)
            Xc = P + T

            if I == 'L':
                k = self.calData['kc_left']
                f = self.calData['fc_left']
                c = self.calData['cc_left']
                alpha = self.calData['alpha_c_left']
            elif I == 'R':
                k = self.calData['kc_right']
                f = self.calData['fc_right']
                c = self.calData['cc_right']
                alpha = self.calData['alpha_c_right']

            inv_z = 1 / Xc[2, :]
            a = Xc[0, :] * inv_z
            b = Xc[1, :] * inv_z
            x = np.vstack([[a], [b]])
            xd2 = self.applyDistortion(x, k)

            xdt = xd2[0, :] + alpha * xd2[1, :]
            xdb = [xd2[1, :]]
            xd3 = np.vstack([xdt, xdb])
            xp = xd3 * f + c

        return xp

    def computeError(self, xL, xpL, xR, xpR):

        e1 = (xL[0, :] - xpL[0, :])**2
        e2 = (xL[1, :] - xpL[1, :])**2
        e3 = (xR[0, :] - xpR[0, :])**2
        e4 = (xR[1, :] - xpR[1, :])**2

        eL = np.sqrt(e1 + e2)
        eR = np.sqrt(e3 + e4)

        error = sum(eL + eR) / 4

        return error


# if __name__ == "__main__":
#    import pyStereoComp
#    x=pyStereoComp.pyStereoComp()
#    x.importMatlabself.calData('10052010_8mm_calibration')
#    import numpy as ny
#    xL=np.array([[826,1230],[915,1044]])
#    xR=np.array([[77,439],[908,1034]])
#    (XL, XR)=x.triangulatePoint(xL,xR)
#    pass
