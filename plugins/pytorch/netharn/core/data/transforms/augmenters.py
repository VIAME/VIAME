import numpy as np
import six
from .data.transforms import augmenter_base

try:
    import imgaug
    from imgaug.parameters import (Uniform, Binomial)
except Exception:
    if 0:
        import warnings
        warnings.warn('imgaug is not availble', DeprecationWarning)


def demodata_hsv_image(w=200, h=200):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> # xdoctest: +REQUIRES(module:cv2)
        >>> rgb255 = demodata_hsv_image()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.imshow(rgb255, colorspace='rgb')
        >>> kwplot.show_if_requested()
    """
    import cv2
    hsv = np.zeros((h, w, 3), dtype=np.float32)

    hue = np.linspace(0, 360, num=w)
    hsv[:, :, 0] = hue[None, :]

    sat = np.linspace(0, 1, num=h)
    hsv[:, :, 1] = sat[:, None]

    val = np.linspace(0, 1, num=3)
    parts = []
    for v in val:
        p = hsv.copy()
        p[:, :, 2] = v
        parts.append(p)
    final_hsv = np.hstack(parts)
    rgb01 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    rgb255 = (rgb01 * 255).astype(np.uint8)
    return rgb255


class HSVShift(augmenter_base.ParamatarizedAugmenter):
    """
    Perform random HSV shift on the RGB data.

    MODIFIED FROM LIGHTNET YOLO into imgaug format

    Args:
        hue (Number): Random number between -hue,hue is used to shift the hue.
            The number is specified as a percentage of the available hue space
            (e.g. hue * 255 or hue * 360).
        saturation (Number): Random number between 1,saturation is used to
            scale the saturation; 50% chance to get 1/dSaturation instead of
            dSaturation
        value (Number): Random number between 1,value is used to scale the
            value; 50% chance to get 1/dValue in stead of dValue

        shift_sat (Number, default=0): random shift applied to saturation
        shift_val (Number, default=0): random shift applied to value

    CommandLine:
        python -m netharn.data.transforms.augmenters HSVShift --show

    Example:
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> self = HSVShift(0.1, 1.5, 1.5)
        >>> img = demodata_hsv_image()
        >>> aug = self.augment_image(img)
        >>> det = self.to_deterministic()
        >>> assert np.all(det.augment_image(img) == det.augment_image(img))
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> import ubelt as ub
        >>> kwplot.autompl()
        >>> kwplot.figure(doclf=True, fnum=3)
        >>> self = HSVShift(0.5, 1.5, 1.5, shift_sat=1.0, shift_val=1.0)
        >>> pnums = kwplot.PlotNums(5, 5)
        >>> #random_state = self.random_state
        >>> import kwarray
        >>> self.reseed(kwarray.ensure_rng(0))
        >>> kwplot.imshow(img, colorspace='rgb', pnum=pnums[0], title='orig')
        >>> for i in range(1, len(pnums)):
        >>>     aug = self.augment_image(img)
        >>>     title = 'aug: {}'.format(ub.repr2(self._prev_params, nl=0, precision=3))
        >>>     kwplot.imshow(aug, colorspace='rgb', pnum=pnums[i], title=title)
        >>> kwplot.show_if_requested()

    Ignore:
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> from .data.transforms.augmenters import *
        >>> lnpre = ub.import_module_from_path(ub.expandpath('~/code/lightnet/lightnet/data/transform/_preprocess.py'))
        >>> self = lnpre.HSVShift(0.1, 1.5, 1.5)
        >>> from PIL import Image
        >>> img = demodata_hsv_image()
        >>> from_ = ub.identity
        >>> #img = Image.fromarray(img)
        >>> #from_ = np.array
        >>> aug = self(img)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> import ubelt as ub
        >>> kwplot.autompl()
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> import random
        >>> random.seed(0)
        >>> pnums = kwplot.PlotNums(5, 5)
        >>> kwplot.imshow(from_(img), colorspace='rgb', pnum=pnums[0], title='orig')
        >>> for i in range(1, len(pnums)):
        >>>     aug = self(img)
        >>>     #title = 'aug: {}'.format(ub.repr2(self._prev_params, nl=0, precision=3))
        >>>     title = 'foo'
        >>>     kwplot.imshow(from_(aug), colorspace='rgb', pnum=pnums[i], title=title)
        >>> kwplot.show_if_requested()
    """
    def __init__(self, hue, sat, val, shift_sat=0, shift_val=0,
                 input_colorspace='rgb'):
        super(HSVShift, self).__init__()
        self.input_colorspace = input_colorspace
        self.hue = Uniform(-hue, hue)
        self.sat = Uniform(1, sat)
        self.val = Uniform(1, val)

        self.shift_sat = Uniform(-shift_sat, shift_sat)
        self.shift_val = Uniform(-shift_val, shift_val)

        self.flip_val = Binomial(.5)
        self.flip_sat = Binomial(.5)

    def _augment_heatmaps(self, heatmaps_on_images, *args, **kw):
        return heatmaps_on_images

    def _augment_images(self, images, random_state, parents, hooks):
        return [self.forward(img, random_state) for img in images]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def forward(self, img, random_state=None):
        import cv2
        assert self.input_colorspace == 'rgb'
        assert img.dtype.kind == 'u' and img.dtype.itemsize == 1

        dh = self.hue.draw_sample(random_state)
        ds = self.sat.draw_sample(random_state)
        dv = self.val.draw_sample(random_state)

        shift_s = self.shift_sat.draw_sample(random_state)
        shift_v = self.shift_val.draw_sample(random_state)

        if self.flip_sat.draw_sample(random_state):
            ds = 1.0 / ds
        if self.flip_val.draw_sample(random_state):
            dv = 1.0 / dv

        self._prev_params = (dh, ds, dv)

        # Note the cv2 conversion to HSV does not go into the 0-1 range,
        # instead it goes into (0-360, 0-1, 0-1) for hue, sat, and val.
        img01 = img.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img01, cv2.COLOR_RGB2HSV)

        hue_bound = 360.0
        sat_bound = 1.0
        val_bound = 1.0

        def wrap_hue(new_hue, hue_bound):
            """ This is about 10x faster than using modulus """
            out = new_hue
            out[out >= hue_bound] -= hue_bound
            out[out < 0] += hue_bound
            return out

        # add to hue
        hsv[:, :, 0] = wrap_hue(hsv[:, :, 0] + (hue_bound * dh), hue_bound)
        if shift_s != 0:
            # shift saturation and value
            hsv[:, :, 1] = np.clip(shift_s + hsv[:, :, 1], 0.0, sat_bound)
        if shift_v != 0:
            hsv[:, :, 2] = np.clip(shift_v + hsv[:, :, 2], 0.0, val_bound)
        # scale saturation and value
        hsv[:, :, 1] = np.clip(ds * hsv[:, :, 1], 0.0, sat_bound)
        hsv[:, :, 2] = np.clip(dv * hsv[:, :, 2], 0.0, val_bound)

        img01 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        img255 = (img01 * 255).astype(np.uint8)
        return img255


class Resize(augmenter_base.ParamatarizedAugmenter):
    """
    Transform images and annotations to the right network dimensions.

    Args:
        target_size (tuple): Scale images to this size (w, h) keeping the
        aspect ratio using a letterbox.

    CommandLine:
        xdoctest -m /home/joncrall/code/netharn/netharn/data/transforms/augmenters.py Resize:0

    Example:
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> from .data.transforms.augmenters import *  # NOQA
        >>> import kwimage
        >>> img = demodata_hsv_image()
        >>> box = kwimage.Boxes([[.45, .05, .10, .05], [0., 0.0, .199, .199], [.24, .05, .01, .05]], format='xywh').to_tlbr()
        >>> bboi = box.to_imgaug(shape=tuple(img.shape))
        >>> self = Resize((40, 30))
        >>> aug1  = self.augment_image(img)
        >>> bboi1 = self.augment_bounding_boxes([bboi])[0]
        >>> box1 = kwimage.Boxes.from_imgaug(bboi1)
        >>> assert box1.br_y.max()  > 8
        >>> assert tuple(bboi1.shape[0:2]) == (30, 40)
        >>> assert tuple(aug1.shape[0:2]) == (30, 40)

    Example:
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> from .data.transforms.augmenters import *  # NOQA
        >>> import kwimage
        >>> img = demodata_hsv_image()
        >>> box = kwimage.Boxes([[450, 50, 100, 50], [0.0, 0, 199, 199], [240, 50, 10, 50]], format='xywh').to_tlbr()
        >>> bboi = box.to_imgaug(shape=img.shape)
        >>> imgT = np.ascontiguousarray(img.transpose(1, 0, 2))
        >>> bboiT = box.transpose().to_imgaug(shape=imgT.shape)
        >>> self = Resize((40, 30))
        >>> self2 = Resize((1000, 1000))
        >>> # ---------------------------
        >>> aug1  = self.augment_image(img)
        >>> bboi1 = self.augment_bounding_boxes([bboi])[0]
        >>> aug2  = self.augment_image(imgT)
        >>> bboi2 = self.augment_bounding_boxes([bboiT])[0]
        >>> aug3  = self2.augment_image(img)
        >>> bboi3 = self2.augment_bounding_boxes([bboi])[0]
        >>> aug4  = self2.augment_image(imgT)
        >>> bboi4 = self2.augment_bounding_boxes([bboiT])[0]
        >>> # ---------------------------
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> pnum_ = kwplot.PlotNums(3, 2)
        >>> kwplot.imshow(img, colorspace='rgb', pnum=pnum_(), title='orig')
        >>> kwplot.draw_boxes(kwimage.Boxes.from_imgaug(bboi))
        >>> kwplot.imshow(imgT, colorspace='rgb', pnum=pnum_(), title='origT')
        >>> kwplot.draw_boxes(kwimage.Boxes.from_imgaug(bboiT))
        >>> # ----
        >>> kwplot.imshow(aug1, colorspace='rgb', pnum=pnum_())
        >>> #kwplot.draw_boxes(kwimage.Boxes.from_imgaug(bboi1))
        >>> x = kwimage.Boxes.from_imgaug(bboi1).translate((-0.5, -0.5))
        >>> kwplot.draw_boxes(x)
        >>> kwplot.imshow(aug2, colorspace='rgb', pnum=pnum_())
        >>> x = kwimage.Boxes.from_imgaug(bboi2).translate((-0.5, -0.5))
        >>> kwplot.draw_boxes(x)
        >>> #kwplot.draw_boxes(kwimage.Boxes.from_imgaug(bboi2))
        >>> # ----
        >>> kwplot.imshow(aug3, colorspace='rgb', pnum=pnum_())
        >>> kwplot.draw_boxes(kwimage.Boxes.from_imgaug(bboi3))
        >>> kwplot.imshow(aug4, colorspace='rgb', pnum=pnum_())
        >>> kwplot.draw_boxes(kwimage.Boxes.from_imgaug(bboi4))

    Ignore:
        image = img
        target_size = np.array(self.target_size)
        orig_size = np.array(img.shape[0:2][::-1])
        shift, scale, embed_size = self._letterbox_transform(orig_size,
                                                             target_size)
    """
    def __init__(self, target_size, fill_color=127, mode='letterbox',
                 border='constant', random_state=None):
        import cv2
        super(Resize, self).__init__(random_state=random_state)
        self.target_size = None if target_size is None else np.array(target_size)
        self.mode = mode

        import imgaug.parameters as iap
        if fill_color == imgaug.ALL:
            self.fill_color = iap.Uniform(0, 255)
        else:
            self.fill_color = iap.handle_continuous_param(
                fill_color, "fill_color", value_range=None,
                tuple_to_uniform=True, list_to_choice=True)

        self._cv2_border_type_map = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'linear_ramp': None,
            'maximum': None,
            'mean': None,
            'median': None,
            'minimum': None,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT,
            'wrap': cv2.BORDER_WRAP,
            cv2.BORDER_CONSTANT: cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE: cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT_101: cv2.BORDER_REFLECT_101,
            cv2.BORDER_REFLECT: cv2.BORDER_REFLECT
        }
        if isinstance(border, six.string_types):
            if border == imgaug.ALL:
                border = [k for k, v in self._cv2_border_type_map.items()
                          if v is not None and isinstance(k, six.string_types)]
            else:
                border = [border]
        if isinstance(border, (list, tuple)):
            from imgaug.parameters import Choice
            border = Choice(border)
        self.border = border
        assert self.mode == 'letterbox', 'thats all folks'

    def forward(self, img, random_state=None):
        orig_size = np.array(img.shape[0:2][::-1])
        assert self.mode == 'letterbox', 'thats all folks'
        shift, scale, embed_size = self._letterbox_transform(orig_size,
                                                             self.target_size)
        new_img = self._img_letterbox_apply(img, embed_size, shift,
                                            self.target_size)
        return new_img

    def _augment_bounding_boxes(self, bounding_boxes_on_images, random_state,
                                parents, hooks):
        # Fix for imgaug 0.4.0
        return self._augment_bounding_boxes_as_keypoints(
            bounding_boxes_on_images, random_state, parents, hooks)

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        # Fix for imgaug 0.4.0
        return self._augment_polygons_boxes_as_keypoints(
            polygons_on_images, random_state, parents, hooks)

    def _augment_images(self, images, random_state, parents, hooks):
        self.target_size = None if self.target_size is None else np.array(self.target_size)
        return [self.forward(img, random_state) for img in images]

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        """
        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> # xdoctest: +REQUIRES(module:kwimage)
            >>> import imgaug
            >>> import kwimage
            >>> tlbr = [[0, 0, 10, 10], [1, 2, 8, 9]]
            >>> shape = (20, 40, 3)
            >>> bboi = kwimage.Boxes(tlbr, 'tlbr').to_imgaug(shape)
            >>> bounding_boxes_on_images = [bboi]
            >>> kps_ois = []
            >>> for bbs_oi in bounding_boxes_on_images:
            >>>     kps = []
            >>>     for bb in bbs_oi.bounding_boxes:
            >>>         kps.extend(bb.to_keypoints())
            >>>     kps_ois.append(imgaug.KeypointsOnImage(kps, shape=bbs_oi.shape))
            >>> keypoints_on_images = kps_ois
            >>> self = LetterboxResize((400, 400))
            >>> aug = self.augment_keypoints(keypoints_on_images)
            >>> assert np.all(aug[0].shape == self.target_size[::-1])
        """
        result = []
        target_size = np.array(self.target_size)
        target_shape = target_size[::-1]
        prev_size = None
        for i, keypoints_on_image in enumerate(keypoints_on_images):
            orig_size = (keypoints_on_image.width, keypoints_on_image.height)

            if prev_size != orig_size:
                # Cache previously computed values
                shift, scale, embed_size = self._letterbox_transform(
                    orig_size, target_size)
                prev_size = orig_size

            try:
                xy = keypoints_on_image.to_xy_array()
            except (Exception, AttributeError):
                xy = keypoints_on_image.get_coords_array()
            xy_aug = (xy * scale) + shift

            try:
                new_keypoint = imgaug.KeypointsOnImage.from_xy_array(
                    xy_aug, shape=target_shape)
            except (Exception, AttributeError):
                new_keypoint = imgaug.KeypointsOnImage.from_coords_array(
                    xy_aug, shape=target_shape)
            # Fix bug in imgaug (TODO: report the bug)
            new_keypoint.shape = target_shape
            result.append(new_keypoint)
        return result

    def _img_letterbox_invert(self, img, orig_size, target_size):
        """
        Args:
            img : the image to scale back up
            orig_size : original wh of the image
            target_size : network input wh

        Example:
            >>> # xdoctest: +REQUIRES(module:kwimage)
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> orig_img = demodata_hsv_image(w=100, h=200)
            >>> orig_size = np.array(orig_img.shape[0:2][::-1])
            >>> target_size = (416, 416)
            >>> self = Resize(target_size)
            >>> img = self.forward(orig_img)
            >>> inverted_img = self._img_letterbox_invert(img, orig_size, target_size)
            >>> assert inverted_img.shape == orig_img.shape
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.imshow(orig_img, fnum=1, pnum=(1, 3, 1))
            >>> kwplot.imshow(img, fnum=1, pnum=(1, 3, 2))
            >>> kwplot.imshow(inverted_img, fnum=1, pnum=(1, 3, 3))
        """
        import cv2
        shift, scale, embed_size = self._letterbox_transform(orig_size, target_size)
        top, bot, left, right = self._padding(embed_size, shift, target_size)

        # Undo padding
        h, w = img.shape[0:2]
        unpadded_img = img[top:h - bot, left:w - right]

        sf = orig_size / embed_size
        dsize = tuple(orig_size)
        # Choose INTER_AREA if we are shrinking the image
        interpolation = cv2.INTER_AREA if sf.sum() < 2 else cv2.INTER_CUBIC
        inverted_img = cv2.resize(unpadded_img, dsize, interpolation=interpolation)
        return inverted_img

    def _boxes_letterbox_apply(self, boxes, orig_size, target_size):
        """
        Apply the letterbox transform to these bounding boxes

        """
        shift, scale, embed_size = self._letterbox_transform(orig_size, target_size)
        new_boxes = boxes.scale(scale).translate(shift)
        return new_boxes

    def _boxes_letterbox_invert(self, boxes, orig_size, target_size):
        """
        Undo the letterbox transform for these bounding boxes. Moves
        the box from `target_size` coordinatse (which are probably square)
        to `orig_size` coordinates (which are probably not square).

        Args:
            boxes (kwimage.Boxes) : boxes to rework in `target_size` coordinates
            orig_size : original wh of the image
            target_size : network input wh (i.e. inp_size)

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> # xdoctest: +REQUIRES(module:kwimage)
            >>> target_size = (416, 416)
            >>> orig_size = (1000, 400)
            >>> import kwimage
            >>> cxywh_norm = kwimage.Boxes(np.array([[.5, .5, .2, .2]]), 'cxywh')
            >>> self = Resize(target_size)
            >>> cxywh = self._boxes_letterbox_invert(cxywh_norm, orig_size, target_size)
            >>> cxywh_norm2 = self._boxes_letterbox_apply(cxywh, orig_size, target_size)
            >>> assert np.allclose(cxywh_norm2.data, cxywh_norm.data)
        """
        shift, scale, embed_size = self._letterbox_transform(orig_size, target_size)
        new_boxes = boxes.translate(-shift).scale(1.0 / scale)
        return new_boxes

    def _letterbox_transform(self, orig_size, target_size):
        """
        aspect ratio preserving scaling + extra padding to equal target size

        scale should be applied before the shift.

        Args:
            orig_size : original wh of the image
            target_size : network input wh

        Returns:
            Tuple:
                shift: x,y shift
                scale: w,h scale
                embed_size: innner w,h of unpadded region

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> # xdoctest: +IGNORE_WHITESPACE
            >>> Resize(None)._letterbox_transform([5, 10], [10, 10])
            (array([2, 0]), array([1., 1.]), array([ 5, 10]))
            >>> Resize(None)._letterbox_transform([10, 5], [10, 10])
            (array([0, 2]), array([1., 1.]), array([10,  5]))
        """
        # determine if width or the height should be used as the scale factor.
        orig_size = np.array(orig_size)
        target_size = np.array(target_size)
        fw, fh = orig_size / target_size
        sf = 1 / fw if fw >= fh else 1 / fh

        # Whats the closest integer size we can resize to?
        embed_size = np.round(orig_size * sf).astype(int)
        # Determine how much padding we need for the top/left side
        # Note: the right/bottom side might need an extra pixel of padding
        # depending on rounding issues.
        shift = np.round((target_size - embed_size) / 2).astype(int)

        scale = embed_size / orig_size
        return shift, scale, embed_size

    @staticmethod
    def _padding(embed_size, shift, target_size):
        pad_lefttop = shift
        pad_rightbot = target_size - (embed_size + shift)

        left, top = pad_lefttop
        right, bot = pad_rightbot
        return top, bot, left, right

    def _img_letterbox_apply(self, img, embed_size, shift, target_size):
        import kwimage
        import cv2
        top, bot, left, right = self._padding(embed_size, shift, target_size)

        orig_size = np.array(img.shape[0:2][::-1])
        channels = kwimage.num_channels(img)

        sf = embed_size / orig_size
        dsize = tuple(embed_size)
        # Choose INTER_AREA if we are shrinking the image
        interpolation = cv2.INTER_AREA if sf.sum() < 2 else cv2.INTER_CUBIC
        if any(d < 0 for d in dsize):
            raise ValueError('dsize={} must be non-negative'.format(dsize))
        scaled = cv2.resize(img, dsize, interpolation=interpolation)

        border = self.border.draw_sample()
        cval = self.fill_color.draw_sample()

        if scaled.dtype == 'f':
            value = (float(cval),) * channels
        else:
            value = (int(cval),) * channels

        borderType = self._cv2_border_type_map[border]
        if borderType is None:
            raise ValueError('bad border type border={}, borderType={}'.format(
                border, borderType))

        hwc255 = cv2.copyMakeBorder(scaled, top, bot, left, right,
                                    borderType=borderType,
                                    value=value)
        return hwc255


LetterboxResize = Resize

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.data.transforms.augmenters all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
