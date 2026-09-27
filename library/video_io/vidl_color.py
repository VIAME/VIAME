"""VXL/VIDL's byte YUV conversion, used only by the vidl_ffmpeg compatibility reader.

The integer coefficients and truncation match vidl_color_convert_yuv2rgb.
Chroma samples are replicated according to the source plane dimensions.
"""
import numpy as np

YUV_FORMATS = {"yuv420p", "yuv422p", "yuv444p", "yuv410p", "yuv411p"}


def planar_yuv_to_rgb(y, u, v):
    height, width = y.shape
    def expand(plane):
        values = plane.astype(np.int32) - 128
        if values.shape != y.shape:
            values = values.repeat((height + plane.shape[0] - 1) // plane.shape[0], axis=0)
            values = values.repeat((width + plane.shape[1] - 1) // plane.shape[1], axis=1)
        return values[:height, :width]
    u, v = expand(u), expand(v)
    y = y.astype(np.int32)
    planar = np.empty((3, height, width), dtype=np.uint8)
    planar[0] = np.clip(y + ((1436 * v) >> 10), 0, 255)
    planar[1] = np.clip(y - ((352 * u + 731 * v) >> 10), 0, 255)
    planar[2] = np.clip(y + ((1814 * u) >> 10), 0, 255)
    return planar.transpose(1, 2, 0)
