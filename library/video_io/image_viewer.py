# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Show each frame of a pipeline in a window, on cv2.

`library/video_io/image_viewer_process.cxx` in python. It is the one
process in the tree that opens a GUI: `cv::imshow` is highgui, which
`image_ops` has no counterpart for and should not grow one. So this is the
same trick as the calibration and disparity ports -- the algorithm stays
OpenCV's and only the language changes -- and it is what takes the last
OpenCV out of `library/video_io`.

The annotation reproduces the C++: a white border above and below sized to
the text, the frame number at the top left, and the configured header and
footer centred in their borders. The glyphs are Hershey's in both, because
this calls the same `cv2.putText`.
"""

import logging

import numpy as np

from kwiver.sprokit.pipeline import process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess

logger = logging.getLogger(__name__)

# The C++ used these three for every string it drew.
FONT_SCALE = 1.0
FONT_THICKNESS = 2
TEXT_GREY = (10, 10, 10)
BORDER_WHITE = (255, 255, 255)


class ImageViewer(KwiverProcess):
    """`image_viewer`: display the input image, and delay."""

    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        for name, default, description in (
                ("pause_time", "0",
                 "Interval to pause between frames. 0 means wait for "
                 "keystroke, Otherwise interval is in seconds (float)"),
                ("annotate_image", "false",
                 "Add frame number and other text to display."),
                ("title", "Display window", "Display window title text."),
                ("header", "", "Header text for image display."),
                ("footer", "",
                 "Footer text for image display. Displayed centered at "
                 "bottom of image.")):
            self.add_config_trait(name, name, default, description)
            self.declare_config_using_trait(name)

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait("timestamp", optional)
        self.declare_input_port_using_trait("image", required)

    def _configure(self):
        self._pause_ms = int(float(self.config_value("pause_time")) * 1000.0)
        self._annotate = str(
            self.config_value("annotate_image")).lower() in ("true", "yes", "1")
        self._title = str(self.config_value("title"))
        self._header = str(self.config_value("header"))
        self._footer = str(self.config_value("footer"))

        self._base_configure()

    def _annotate_image(self, image, frame):
        import cv2

        text = "Frame: %d" % frame

        (_, height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)

        bordered = cv2.copyMakeBorder(
            image, height + 8, height + 8, 0, 0, cv2.BORDER_CONSTANT,
            value=BORDER_WHITE)

        cv2.putText(bordered, text, (5, height + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, TEXT_GREY, FONT_THICKNESS)

        for caption, at_top in ((self._header, True), (self._footer, False)):
            if not caption:
                continue

            (width, caption_height), _ = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)

            origin = ((bordered.shape[1] - width) // 2,
                      caption_height + 3 if at_top else bordered.shape[0] - 3)

            cv2.putText(bordered, caption, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE, TEXT_GREY, FONT_THICKNESS)

        return bordered

    def _step(self):
        import cv2

        frame_time = None
        if self.has_input_port_edge_using_trait("timestamp"):
            frame_time = self.grab_input_using_trait("timestamp")

        container = self.grab_from_port_using_trait("image")

        # `vital_to_ocv( ..., BGR_COLOR )` is what the C++ asked the bridge
        # for, which is this swap: the array is RGB and cv2 shows BGR.
        image = np.ascontiguousarray(container.asarray())

        if image.ndim == 3 and image.shape[2] >= 3:
            image = image[:, :, ::-1].copy()

        if self._annotate:
            frame = frame_time.get_frame() if frame_time is not None else -1
            image = self._annotate_image(image, frame)

        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        cv2.imshow(self._title, image)
        cv2.waitKey(self._pause_ms)


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = "python:viame.video_io"

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        "image_viewer", "Display input image and delay", ImageViewer)

    process_factory.mark_process_module_as_loaded(module_name)
