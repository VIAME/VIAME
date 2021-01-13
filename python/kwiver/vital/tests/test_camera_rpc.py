"""
ckwg +31
Copyright 2020 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Tests for CameraRPC interface class.

"""
import unittest
import nose.tools as nt
import numpy as np
import math

from kwiver.vital.tests.py_helpers import no_call_pure_virtual_method
from kwiver.vital.tests.cpp_helpers import camera_rpc_helpers as crpch
from kwiver.vital.types import CameraRPC as crpc, SimpleCameraRPC as srpc



class TestCameraRPC(unittest.TestCase):
    def test_construct(self):
        crpc()

    def test_pure_virts(self):
        no_call_pure_virtual_method(crpc().project, np.array([1.0, 2.0, 10.0]))
        no_call_pure_virtual_method(crpc().back_project, np.array([1.0, 2.0]), 1.0)
        no_call_pure_virtual_method(crpc().rpc_coeffs)
        no_call_pure_virtual_method(crpc().world_scale)
        no_call_pure_virtual_method(crpc().world_offset)
        no_call_pure_virtual_method(crpc().image_scale)
        no_call_pure_virtual_method(crpc().image_offset)
        no_call_pure_virtual_method(crpc().image_height)
        no_call_pure_virtual_method(crpc().image_width)


class TestSimpleCameraRPC(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.m_world_scale = np.array([0, 1, 0])
        self.m_world_offset = np.array([1, 0, 1])
        self.m_image_scale = np.array([1, 1])
        self.m_image_offset = np.array([0, 2])
        self.m_image_width = 1080
        self.m_image_height = 720
        self.m_coeffs = np.ndarray(shape=(4, 20),dtype=float)
        self.m_srpc = srpc(self.m_world_scale, self.m_world_offset, self.m_image_scale,
                                 self.m_image_offset, self.m_coeffs,
                                 self.m_image_width, self.m_image_height)

    def test_constructors(self):
        srpc()
        srpc(self.m_world_scale, self.m_world_offset, self.m_image_scale,
                                 self.m_image_offset, self.m_coeffs,
                                 self.m_image_width, self.m_image_height)
        srpc(srpc())
        srpc(srpc(self.m_world_scale, self.m_world_offset, self.m_image_scale,
                                 self.m_image_offset, self.m_coeffs,
                                 self.m_image_width, self.m_image_height))

    def test_methods(self):
        cameraRPC = srpc()

        # test setters && getters (getters inherited from camera_rpc)
        cameraRPC.set_rpc_coeffs(self.m_coeffs)
        np.testing.assert_array_equal(self.m_coeffs, cameraRPC.rpc_coeffs())
        cameraRPC.set_world_scale(self.m_world_scale)
        np.testing.assert_array_equal(self.m_world_scale, cameraRPC.world_scale())
        cameraRPC.set_world_offset(self.m_world_offset)
        np.testing.assert_array_equal(self.m_world_offset, cameraRPC.world_offset())
        cameraRPC.set_image_scale(self.m_image_scale)
        np.testing.assert_array_equal(self.m_image_scale, cameraRPC.image_scale())
        cameraRPC.set_image_offset(self.m_image_offset)
        np.testing.assert_array_equal(self.m_image_offset, cameraRPC.image_offset())
        cameraRPC.set_image_width(self.m_image_width)
        np.testing.assert_array_equal(self.m_image_width, cameraRPC.image_width())
        cameraRPC.set_image_height(self.m_image_height)
        np.testing.assert_array_equal(self.m_image_height, cameraRPC.image_height())

        # test project, back_project, jacobian
        proj_2d = srpc().project(np.array([1.0, 2.0, 10.0]))
        np.testing.assert_array_equal(proj_2d, np.array([1.0, 2.0]))
        proj_3d = srpc().back_project([1.0, 2.0], 1.0)
        np.testing.assert_array_equal(np.array([1.0, 2.0 ,1.0]), proj_3d)
        ret_val = np.array([0, 0])
        ret_jac = np.ndarray(shape=(2, 2), dtype=(float), order='F')
        ret_jac.setflags(write=1)
        rpc_coeffs_buffer = [[0.006585953, -1.032582, 0.001740937, 0.03034485, 0.0008819178, -0.000167943, 0.0001519299 -0.00626254, -0.00107337, 9.099077e-06, 2.608985e-06, -2.947004e-05, 2.231277e-05, 4.587831e-06, 4.16379e-06, 0.0003464555, 3.598323e-08, -2.859541e-06, 5.159311e-06, -1.349187e-07],
         [1, 0.0003374458, 0.0008965622, -0.0003730697, -2.666499e-05, -2.711356e-06, 5.454434e-07, 4.485658e-07, 2.534922e-05, -4.546709e-06, 0, -1.056044e-07, -5.626866e-07, 2.243313e-08, -2.108053e-07, 9.199534e-07, 0, -3.887594e-08, -1.437016e-08, 0],
         [0.0002703625, 0.04284488, 1.046869, 0.004713542, -0.0001706129, -1.525177e-07, 1.255623e-05, -0.0005820134,  -0.000710512, -2.510676e-07, 3.179984e-06, 3.120413e-06, 3.19923e-05, 4.194369e-06, 7.475295e-05, 0.0003630791, 0.0001021649, 4.493725e-07, 3.156566e-06, 4.596505e-07],
         [1, 0.0001912806, 0.0005166397, -1.45044e-05, -3.860133e-05, 2.634582e-06, -4.551145e-06, 6.859296e-05, -0.0002410782, 9.753265e-05, -1.456261e-07, 5.310624e-08, -1.913253e-05, 3.18203e-08, 3.870586e-07, -0.000206842, 9.128349e-08, 0, -2.506197e-06, 0]]

        rpc_coeffs = np.ndarray(shape = (4,20), dtype = float)
        for x in range(len(rpc_coeffs_buffer)):
            for y in range(len(rpc_coeffs_buffer[x])):
                rpc_coeffs[x][y] = rpc_coeffs_buffer[x][y]
        world_scale = np.array([0.0928, 0.0708, 501])
        world_offset = np.array([-58.6096, -34.4732, 31])
        image_scale = np.array([21250, 21478])
        image_offset = np.array([21249, 21477])
        image_width = 1000
        image_height = 1000
        s = srpc(world_scale, world_offset, image_scale, image_offset, rpc_coeffs, image_height, image_width)
        ret_jac, ret_val = s.jacobian(np.array([-58.58, -34.49, 20.92]), ret_jac, ret_val)

        self.assertFalse(ret_jac.flags.owndata)
        np.testing.assert_array_almost_equal(ret_jac[1], np.array([ 2.92836539e-002, -2.25968e-001]))

        # test clone
        cameraRPC_2 = self.m_srpc.clone()
        # test for slice on c++ side
        self.assertIsInstance(cameraRPC_2, srpc)
        # test for value copying
        np.testing.assert_array_equal(self.m_coeffs, cameraRPC_2.rpc_coeffs())
        np.testing.assert_array_equal(self.m_world_scale, cameraRPC_2.world_scale())
        np.testing.assert_array_equal(self.m_world_offset, cameraRPC_2.world_offset())
        np.testing.assert_array_equal(self.m_image_scale, cameraRPC_2.image_scale())
        np.testing.assert_array_equal(self.m_image_offset, cameraRPC_2.image_offset())
        np.testing.assert_array_equal(self.m_image_width, cameraRPC_2.image_width())
        np.testing.assert_array_equal(self.m_image_height, cameraRPC_2.image_height())

    def test_call_overrides(self):
        # clone
        clone_ = crpch.call_clone(self.m_srpc)
        self.assertIsInstance(clone_, srpc)
        np.testing.assert_array_equal(self.m_coeffs, clone_.rpc_coeffs())
        np.testing.assert_array_equal(self.m_world_scale, clone_.world_scale())
        np.testing.assert_array_equal(self.m_world_offset, clone_.world_offset())
        np.testing.assert_array_equal(self.m_image_scale, clone_.image_scale())
        np.testing.assert_array_equal(self.m_image_offset, clone_.image_offset())
        np.testing.assert_array_equal(self.m_image_width, clone_.image_width())
        np.testing.assert_array_equal(self.m_image_height, clone_.image_height())

        # other members
        np.testing.assert_array_equal(self.m_coeffs, crpch.call_rpc_coeffs(self.m_srpc))
        np.testing.assert_array_equal(self.m_world_scale, crpch.call_world_scale(self.m_srpc))
        np.testing.assert_array_equal(self.m_world_offset, crpch.call_world_offset(self.m_srpc))
        np.testing.assert_array_equal(self.m_image_width, crpch.call_image_width(self.m_srpc))
        np.testing.assert_array_equal(self.m_image_height, crpch.call_image_height(self.m_srpc))
        np.testing.assert_array_equal(self.m_image_scale, crpch.call_image_scale(self.m_srpc))
        np.testing.assert_array_equal(self.m_image_offset, crpch.call_image_offset(self.m_srpc))
        np.testing.assert_array_equal(np.array([1.0, 2.0]), crpch.call_project(srpc(), [1.0, 2.0, 10.0]))
        np.testing.assert_array_equal(np.array([1.0, 2.0, 1.0]), crpch.call_back_project(srpc(), np.array([1.0, 2.0]), 1.0))

class InheritedRPC(crpc):
    def __init__(self):
        crpc.__init__(self)

    def clone(self):
        return InheritedRPC()

    def rpc_coeffs(self):
        return np.ndarray(shape=(4, 20), dtype=float)

    def world_offset(self):
        return np.array([1, 8, 9])

    def world_scale(self):
        return np.array([8, 9, 10])

    def image_scale(self):
        return np.array([0.5, 2.9])

    def image_offset(self):
        return np.array([1.0, 0.75])

    def image_width(self):
        return 3840

    def image_height(self):
        return 2160

    def project(self, pt):
        return np.array([99, 100])

    def back_project(self, pt, elev):
        return np.array([1, 2, 3])


class TestInheritedRPC(unittest.TestCase):
    def test_construct(self):
        InheritedRPC()

    def test_inheritance(self):
        nt.ok_(issubclass(InheritedRPC, crpc))

    def test_clone(self):
        irpc = InheritedRPC()
        cloned_ = crpch.call_clone(irpc)
        self.assertIsInstance(cloned_, InheritedRPC)
        self.assertEqual(cloned_.image_width(), InheritedRPC().image_width())

    def test_methods(self):
        irpc = InheritedRPC()
        self.assertEqual(crpch.call_rpc_coeffs(irpc).shape, np.ndarray(shape=(4, 20), dtype=float).shape)
        np.testing.assert_array_equal(crpch.call_world_scale(irpc), np.array([8, 9, 10]))
        np.testing.assert_array_equal(crpch.call_world_offset(irpc), np.array([1, 8, 9]))
        np.testing.assert_array_equal(crpch.call_image_scale(irpc), np.array([0.5, 2.9]))
        np.testing.assert_array_equal(crpch.call_image_offset(irpc), np.array([1.0, 0.75]))
        self.assertEqual(crpch.call_image_width(irpc), 3840)
        self.assertEqual(crpch.call_image_height(irpc), 2160)
        np.testing.assert_array_equal(crpch.call_project(irpc, np.array([1, 2, 3])), np.array([99, 100]))
        np.testing.assert_array_equal(crpch.call_back_project(irpc, np.array([0, 0.125]), 1), np.array([1, 2, 3]))
