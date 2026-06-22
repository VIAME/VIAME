# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
KWIVER algorithm for stereo depth/disparity estimation using NVIDIA
Fast-Foundation-Stereo, served from the SINGLE-FILE ONNX export (or the
matching TensorRT engine).

This sibling of fast_foundation_stereo.py exists because the ONNX/TRT
export has a different runtime contract from the PyTorch checkpoint:

  - Inputs are ImageNet-normalised RGB tensors (NCHW float32).
  - Inputs are resized to the model's baked-in resolution (read from the
    sidecar cfg.yaml, e.g. 576x960 or 320x736). Disparity is rescaled by
    the inverse width ratio so the returned map matches the original
    image resolution.
  - I/O bindings are named: 'left_image' / 'right_image' -> 'disparity'.

ONNX Runtime is the default backend; pass an .engine file (or a directory
containing one) to use TensorRT instead.
"""

import os
import json
import numpy as np

import scriptconfig as scfg

from kwiver.vital.algo import ComputeStereoDepthMap
from kwiver.vital.types import Image, ImageContainer

from viame.core.utils import str2bool

from viame.pytorch.utilities import vital_config_update


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FastFoundationStereoOnnxConfig(scfg.DataConfig):
    """
    Configuration for :class:`FastFoundationStereoOnnx`.
    """
    model_path = scfg.Value(
        '', help='Path to a single-file fast-foundation-stereo .onnx model '
                 '(or a .engine if using TensorRT).')
    config_path = scfg.Value(
        '', help='Path to the sidecar .yaml describing the export '
                 '(image_size, etc.). When empty, looks for <model>.yaml '
                 'next to the model file.')
    backend = scfg.Value(
        'auto', help="'auto' (pick from extension), 'onnxruntime', or 'tensorrt'.")
    device = scfg.Value(
        'auto', help="'auto', 'cuda', 'cuda:N', or 'cpu' (only meaningful "
                     "for the onnxruntime backend; tensorrt always uses CUDA).")
    output_mode = scfg.Value(
        'disparity', help="'disparity' (scaled *256, uint16) or 'depth' "
                          "(millimeters, uint16). 'depth' requires "
                          "calibration_file.")
    calibration_file = scfg.Value(
        '', help='Path to KWIVER stereo calibration file (JSON) - required '
                 'for depth output.')
    remove_invisible = scfg.Value(
        True, help='Set invalid disparity (negative x in right image) to '
                  'infinity.')


class FastFoundationStereoOnnx(ComputeStereoDepthMap):
    """ComputeStereoDepthMap impl for Fast-Foundation-Stereo ONNX/TRT."""

    def __init__(self):
        ComputeStereoDepthMap.__init__(self)

        self._config = FastFoundationStereoOnnxConfig()

        self._focal_length = 0.0
        self._baseline = 0.0
        self._principal_x = 0.0
        self._principal_y = 0.0

        self._runner = None
        self._target_h = 0
        self._target_w = 0
        self._input_left_name = 'left_image'
        self._input_right_name = 'right_image'
        self._output_disp_name = 'disparity'

    def get_configuration(self):
        cfg = super(ComputeStereoDepthMap, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    def set_configuration(self, cfg_in):
        import yaml

        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config['remove_invisible'] = str2bool(self._config['remove_invisible'])

        model_path = self._config['model_path']
        if not model_path:
            raise RuntimeError("model_path must be specified")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")

        # Sidecar yaml: explicit, or <model>.yaml, or 'config.yaml' / 'onnx.yaml'
        # in the same dir. Mirrors run_demo_single_trt.py's resolve_config.
        config_path = self._config['config_path']
        if not config_path:
            config_path = self._find_sidecar_yaml(model_path)
        if not config_path or not os.path.exists(config_path):
            raise RuntimeError(
                f"Could not find sidecar .yaml for {model_path}. "
                f"Set config_path explicitly.")

        with open(config_path, 'r') as ff:
            yaml_cfg = yaml.safe_load(ff) or {}
        image_size = yaml_cfg.get('image_size')
        if not image_size or len(image_size) != 2:
            raise RuntimeError(
                f"sidecar yaml at {config_path} missing image_size: [H, W]")
        self._target_h, self._target_w = int(image_size[0]), int(image_size[1])

        calibration_file = self._config['calibration_file']
        if calibration_file and os.path.exists(calibration_file):
            self._load_calibration(calibration_file)

        if self._config['output_mode'] == 'depth':
            if self._focal_length <= 0 or self._baseline <= 0:
                raise RuntimeError(
                    "calibration_file with valid focal length and baseline "
                    "required for depth output")

        # Pick backend
        backend = self._config['backend']
        if backend == 'auto':
            backend = 'tensorrt' if model_path.endswith('.engine') else 'onnxruntime'

        if backend == 'onnxruntime':
            self._runner = self._build_ort_runner(model_path)
        elif backend == 'tensorrt':
            self._runner = self._build_trt_runner(model_path)
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("model_path"):
            print("Error: model_path is required")
            return False
        model_path = str(cfg.get_value("model_path"))
        if not model_path:
            print("Error: model_path must not be empty")
            return False

        output_mode = str(cfg.get_value("output_mode"))
        if output_mode not in ['disparity', 'depth']:
            print(f"Error: output_mode must be 'disparity' or 'depth', "
                  f"got '{output_mode}'")
            return False

        return True

    def _find_sidecar_yaml(self, model_path):
        model_dir = os.path.dirname(model_path)
        base = os.path.splitext(os.path.basename(model_path))[0]
        for cand in (
            os.path.join(model_dir, base + '.yaml'),
            os.path.join(model_dir, 'config.yaml'),
            os.path.join(model_dir, 'onnx.yaml'),
        ):
            if os.path.exists(cand):
                return cand
        return ''

    def _build_ort_runner(self, onnx_path):
        import onnxruntime as ort

        device = self._config['device']
        avail = ort.get_available_providers()
        providers = []
        if device == 'cpu':
            providers.append('CPUExecutionProvider')
        else:
            if 'CUDAExecutionProvider' in avail:
                # device_id from 'cuda' or 'cuda:N'
                device_id = 0
                if ':' in device:
                    try:
                        device_id = int(device.split(':', 1)[1])
                    except ValueError:
                        device_id = 0
                providers.append((
                    'CUDAExecutionProvider',
                    {'device_id': device_id},
                ))
            providers.append('CPUExecutionProvider')

        session = ort.InferenceSession(onnx_path, providers=providers)

        in_names = [inp.name for inp in session.get_inputs()]
        out_names = [out.name for out in session.get_outputs()]

        # Discover bindings by name fallback. Default is left_image / right_image.
        if self._input_left_name not in in_names or \
                self._input_right_name not in in_names:
            # Fallback: assume input order is left, right.
            if len(in_names) >= 2:
                self._input_left_name, self._input_right_name = in_names[0], in_names[1]
            else:
                raise RuntimeError(
                    f"ONNX model has unexpected inputs: {in_names}")
        if self._output_disp_name not in out_names:
            self._output_disp_name = out_names[0]

        class OrtRunner:
            def __init__(self, sess, in_l, in_r, out_d):
                self.sess = sess
                self.in_l = in_l
                self.in_r = in_r
                self.out_d = out_d

            def run(self, left_chw, right_chw):
                feed = {self.in_l: left_chw, self.in_r: right_chw}
                outputs = self.sess.run([self.out_d], feed)
                return outputs[0]

        return OrtRunner(
            session, self._input_left_name,
            self._input_right_name, self._output_disp_name)

    def _build_trt_runner(self, engine_path):
        import torch
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize TRT engine from {engine_path}; "
                f"likely TensorRT version mismatch (have "
                f"{trt.__version__}). Rebuild the engine via trtexec.")
        context = engine.create_execution_context()

        def trt_to_torch_dtype(dt):
            if dt == trt.DataType.FLOAT:
                return torch.float32
            if dt == trt.DataType.HALF:
                return torch.float16
            if dt == trt.DataType.INT32:
                return torch.int32
            raise RuntimeError(f"Unsupported TRT dtype: {dt}")

        # Cache binding names
        n = engine.num_io_tensors
        names = [engine.get_tensor_name(i) for i in range(n)]
        in_names = [
            n_ for n_ in names
            if engine.get_tensor_mode(n_) == trt.TensorIOMode.INPUT]
        out_names = [
            n_ for n_ in names
            if engine.get_tensor_mode(n_) == trt.TensorIOMode.OUTPUT]
        if 'left_image' not in in_names or 'right_image' not in in_names:
            if len(in_names) >= 2:
                self._input_left_name = in_names[0]
                self._input_right_name = in_names[1]
            else:
                raise RuntimeError(
                    f"TRT engine has unexpected inputs: {in_names}")
        if 'disparity' not in out_names:
            self._output_disp_name = out_names[0]

        in_l = self._input_left_name
        in_r = self._input_right_name
        out_d = self._output_disp_name

        class TrtRunner:
            def __init__(self):
                self.engine = engine
                self.context = context

            def run(self, left_chw, right_chw):
                left_t = torch.from_numpy(left_chw).cuda()
                right_t = torch.from_numpy(right_chw).cuda()

                tensors = {in_l: left_t, in_r: right_t}
                for nm, t in list(tensors.items()):
                    expected = trt_to_torch_dtype(
                        self.engine.get_tensor_dtype(nm))
                    if t.dtype != expected:
                        tensors[nm] = t.to(expected)
                    if not tensors[nm].is_contiguous():
                        tensors[nm] = tensors[nm].contiguous()
                    self.context.set_input_shape(nm, tuple(tensors[nm].shape))

                outputs = {}
                for nm in out_names:
                    shp = tuple(self.context.get_tensor_shape(nm))
                    dt = trt_to_torch_dtype(self.engine.get_tensor_dtype(nm))
                    outputs[nm] = torch.empty(shp, device='cuda', dtype=dt)

                for nm, t in tensors.items():
                    self.context.set_tensor_address(nm, int(t.data_ptr()))
                for nm, t in outputs.items():
                    self.context.set_tensor_address(nm, int(t.data_ptr()))

                stream = torch.cuda.current_stream().cuda_stream
                ok = self.context.execute_async_v3(stream)
                assert ok, "TRT execute_async_v3 returned failure"
                return outputs[out_d].float().cpu().numpy()

        return TrtRunner()

    def _load_calibration(self, cal_fpath):
        with open(cal_fpath, 'r') as f:
            data = json.load(f)

        self._focal_length = float(data.get('fx_left', 0.0))
        self._principal_x = float(data.get('cx_left', 0.0))
        self._principal_y = float(data.get('cy_left', 0.0))

        T = data.get('T', [0.0, 0.0, 0.0])
        if isinstance(T, list) and len(T) >= 3:
            self._baseline = abs(T[0])
            if self._baseline < 1e-6:
                self._baseline = float(np.sqrt(T[0]**2 + T[1]**2 + T[2]**2))
        else:
            self._baseline = 0.0

        print(f"Loaded calibration: focal_length={self._focal_length}, "
              f"baseline={self._baseline}, principal=({self._principal_x}, "
              f"{self._principal_y})")

    def _format_image(self, image_container):
        img_npy = image_container.image().asarray().astype('uint8')
        if len(img_npy.shape) == 2:
            img_npy = np.stack((img_npy,) * 3, axis=-1)
        elif img_npy.shape[2] == 1:
            img_npy = np.concatenate([img_npy] * 3, axis=-1)
        return img_npy

    def compute(self, left_image, right_image):
        import cv2

        left_npy = self._format_image(left_image)
        right_npy = self._format_image(right_image)

        if left_npy.shape != right_npy.shape:
            raise RuntimeError(
                f"Left and right image dimensions must match: "
                f"{left_npy.shape} vs {right_npy.shape}")

        H_orig, W_orig = left_npy.shape[:2]
        target_h, target_w = self._target_h, self._target_w

        # Resize to model resolution by direct stretch — matches the
        # reference run_demo_single_trt.py path. Disparity is later
        # rescaled by 1/fx so the returned map covers the original
        # resolution.
        fx = target_w / float(W_orig)
        if (target_h, target_w) != (H_orig, W_orig):
            left_npy = cv2.resize(
                left_npy, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            right_npy = cv2.resize(
                right_npy, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # ImageNet normalisation, NCHW float32
        left_norm = (left_npy.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        right_norm = (right_npy.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        left_chw = np.transpose(left_norm, (2, 0, 1))[None]
        right_chw = np.transpose(right_norm, (2, 0, 1))[None]
        left_chw = np.ascontiguousarray(left_chw)
        right_chw = np.ascontiguousarray(right_chw)

        disp = self._runner.run(left_chw, right_chw)
        disp = np.asarray(disp).reshape(target_h, target_w).clip(0, None)

        # Scale disparity back to original image coordinates.
        if (target_h, target_w) != (H_orig, W_orig):
            disp = cv2.resize(
                disp, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            disp = disp / fx  # disparity scales inversely with image scale

        if self._config['remove_invisible']:
            yy, xx = np.meshgrid(
                np.arange(H_orig), np.arange(W_orig), indexing='ij')
            invalid = (xx - disp) < 0
            disp[invalid] = np.inf

        if self._config['output_mode'] == 'depth':
            safe_disp = np.where(disp > 0, disp, 1e-6)
            depth_npy = (self._focal_length * self._baseline) / safe_disp
            depth_npy = np.where(disp > 0, depth_npy, 0)
            depth_mm = (depth_npy * 1000).clip(0, 65535).astype(np.uint16)
            return ImageContainer(Image(depth_mm))
        else:
            disp_output = (disp * 256).clip(0, 65535).astype(np.uint16)
            return ImageContainer(Image(disp_output))


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "fast_foundation_stereo_onnx"

    if algorithm_factory.has_algorithm_impl_name(
            FastFoundationStereoOnnx.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "Stereo depth/disparity estimation using NVIDIA Fast-Foundation-Stereo "
        "ONNX/TensorRT export",
        FastFoundationStereoOnnx
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
