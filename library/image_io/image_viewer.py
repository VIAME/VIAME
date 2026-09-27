# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Display pipeline images with Pillow annotations and Tk windows."""

import logging

import numpy as np

from viame.pipeline import process
from viame.processes.base import ViameProcess

logger = logging.getLogger(__name__)

# The C++ used these three for every string it drew.
FONT_SCALE = 1.0
FONT_THICKNESS = 2
TEXT_GREY = (10, 10, 10)
BORDER_WHITE = (255, 255, 255)


class ImageViewer(ViameProcess):
    """`image_viewer`: display the input image, and delay."""

    def __init__(self, conf):
        ViameProcess.__init__(self, conf)

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
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default()
        text = "Frame: %d" % frame
        height = 28
        bordered = Image.new("RGB", (image.shape[1], image.shape[0] + 2*height), BORDER_WHITE)
        bordered.paste(Image.fromarray(image), (0,height))
        draw = ImageDraw.Draw(bordered)
        draw.text((5,3), text, font=font, fill=TEXT_GREY)
        for caption, at_top in ((self._header,True),(self._footer,False)):
            if caption:
                width = draw.textlength(caption,font=font)
                draw.text(((bordered.width-width)//2,3 if at_top else bordered.height-height+3),
                          caption,font=font,fill=TEXT_GREY)
        return np.asarray(bordered).copy()

    def _step(self):
        from viame.image_io.display import show

        frame_time = None
        if self.has_input_port_edge_using_trait("timestamp"):
            frame_time = self.grab_input_using_trait("timestamp")

        container = self.grab_from_port_using_trait("image")

        image = np.ascontiguousarray(container.asarray())

        if self._annotate:
            frame = frame_time.get_frame() if frame_time is not None else -1
            image = self._annotate_image(image, frame)

        show(image, title=self._title, delay_ms=self._pause_ms)


def __sprokit_register__():
    from viame.pipeline import process_factory

    module_name = "python:viame.image_io"

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        "image_viewer", "Display input image and delay", ImageViewer)

    process_factory.mark_process_module_as_loaded(module_name)
