import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        '''
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        '''
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(F.pad(x, (padding3, padding4, padding1, padding2), mode='replicate'),
                         self.size, stride=self.stride)
        return x


class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
            expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride).contiguous().\
            view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)

        return x


class Upsample_interpolate(nn.Module):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        out = F.interpolate(x, size=(x.size(2) * self.stride, x.size(3) * self.stride), mode='nearest')
        return out


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        # upstream uses "/", which yields floats that view() rejects
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


# for route, shortcut and sam
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


class YoloLayer(nn.Module):
    """
    YOLO output layer defined in
    https://github.com/AlexeyAB/darknet/blob/master/src/yolo_layer.c#L1087
    """
    def __init__(
            self,
            anchor_mask=[],
            num_classes=0,
            anchors=[],
            num_anchors=1,
            stride=32,
            model_out=False
            ):
        super(YoloLayer, self).__init__()
        self.net_height, self.net_width = (None, None)
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1
        self.new_coords = False
        self.model_out = model_out

    def create_anchors(self, n_anchors_x, n_anchors_y):
        """Create the anchors bboxes (cx, cy, w, h) from the pre-defined grid."""
        num_anchors_per_mask = n_anchors_x*n_anchors_y

        anchors_y, anchors_x = torch.meshgrid(
            torch.arange(n_anchors_x, dtype=torch.float32),
            torch.arange(n_anchors_y, dtype=torch.float32))
        # match boxes dimensions: [ny, nx] -> [ny*nx*3]
        anchors_x = (anchors_x.flatten()).repeat(3)
        anchors_y = (anchors_y.flatten()).repeat(3)

        # anchors width and height definied in the "anchors" list and then masked
        anchors_masked_list = []
        anchors_tensor = torch.tensor(self.anchors, requires_grad=False).reshape(-1, 2)
        for anchor_mask in self.anchor_mask:
            anchors_masked_list.append(
                anchors_tensor[anchor_mask:(anchor_mask+1)].repeat(num_anchors_per_mask, 1))
        anchors_masked = torch.cat(anchors_masked_list, dim=0)
        anchors_w = anchors_masked[:, 0].flatten()
        anchors_h = anchors_masked[:, 1].flatten()

        return anchors_x, anchors_y, anchors_w, anchors_h

    def forward(self, output, target=None):
        if self.training:
            return output

        # n_dims is bboxes coords and confs (xywhcp*n)
        # with n anchor types (anchor_mask), c objectness and p classes probs
        n_dims = output.size(1)
        n_batch = output.size(0)
        conv_height = output.size(2)  # last conv height, or the anchors in y dir (vertical)
        conv_width = output.size(3)  # last conv width, or the anchors in x dir (horizontal)
        num_anchors = len(self.anchor_mask)

        ####
        # Get all components cxy, wh, probabilities and confidences (i.e. objectness)
        # from previous (conv) layer output
        ####
        #TODO optimize
        # output.reshape(1, 3, 8, -1)  # decomposing anchor types in 2d dim
        # test.permute((0, 2, 1, 3)).reshape(1, 8, -1)  # bbox coors and label in 2d dim
        bxy_list = []
        bwh_list = []
        confs_list = []
        probs_list = []
        for i in range(num_anchors):
            begin = i * (5 + self.num_classes)
            end = (i + 1) * (5 + self.num_classes)
            bxy_list.append(output[:, begin:(begin+2)].reshape(n_batch, 2, -1))
            bwh_list.append(output[:, (begin+2):(begin+4)].reshape(n_batch, 2, -1))
            confs_list.append(output[:, (begin+4):(begin+5)].reshape(n_batch, 1, -1))
            probs_list.append(output[:, (begin+5):end].reshape(n_batch, self.num_classes, -1))
        # [batch, d * num_anchors, H, W] -> [batch, d, num_anchors * H * W]
        # where d is the num of dimension (for ex: num_classes)
        bxy = torch.transpose(torch.cat(bxy_list, dim=-1), 1, 2)
        bwh = torch.transpose(torch.cat(bwh_list, dim=-1), 1, 2)
        confs = torch.transpose(torch.cat(confs_list, dim=-1), 1, 2)
        probs = torch.transpose(torch.cat(probs_list, dim=-1), 1, 2)

        #####
        # rescale bxy
        #####
        bxy = bxy * self.scale_x_y - 0.5 * (self.scale_x_y - 1)

        ####
        # get (cx, cy, w, h) from pre-defined anchors on a grid
        ###
        #TODO: could be done before forward
        anchors_x, anchors_y, anchors_w, anchors_h = self.create_anchors(
            conv_height, conv_width)
        # for x and y we just add the grid values
        bx = bxy[..., 0] + anchors_x
        by = bxy[..., 1] + anchors_y
        # add a non-linear transforms for w and h
        # https://github.com/AlexeyAB/darknet/issues/7645#issuecomment-826670205
        if self.new_coords:
            bw = bwh[..., 0]**2 * 4 * anchors_w
            bh = bwh[..., 1]**2 * 4 * anchors_h
        else:
            bw = torch.exp(bwh[..., 0]) * 4 * anchors_w
            bh = torch.exp(bwh[..., 1]) * 4 * anchors_h

        #####
        # Normalize bboxes
        #####
        bx = torch.unsqueeze(bx / conv_width, dim=-1)
        by = torch.unsqueeze(by / conv_height, dim=-1)
        bw = torch.unsqueeze(bw / self.net_width, dim=-1)
        bh = torch.unsqueeze(bh / self.net_height, dim=-1)
        #TODO: torch.stack((bx, by, bw, bh), axis=-1) and remove unsqueeze
        boxes = torch.cat([bx, by, bw, bh], dim=-1)

        return boxes, probs, confs
