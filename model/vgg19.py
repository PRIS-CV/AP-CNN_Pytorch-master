import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import time
import os
import numpy as np
import cv2
import random
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import init
from torch.utils.checkpoint import checkpoint_sequential

def get_merge_bbox(dets, inds):
    xx1 = np.min(dets[inds][:,0])
    yy1 = np.min(dets[inds][:,1])
    xx2 = np.max(dets[inds][:,2])
    yy2 = np.max(dets[inds][:,3])

    return np.array((xx1, yy1, xx2, yy2))

def pth_nms_merge(dets, thresh, topk):
    dets = dets.cpu().data.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    boxes_merge = []
    cnt = 0
    while order.size > 0:
        i = order[0]
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]

        inds_merge = np.where((ovr > 0.5)*(0.9*scores[i]<scores[order[1:]]))[0]
        boxes_merge.append(get_merge_bbox(dets, np.append(i, order[inds_merge+1])))
        order = order[inds + 1]

        cnt += 1
        if cnt >= topk:
            break

    return torch.from_numpy(np.array(boxes_merge))

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SimpleFPA(nn.Module):
    def __init__(self, in_planes, out_planes):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(SimpleFPA, self).__init__()

        self.channels_cond = in_planes
        # Master branch
        self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

        # Global pooling branch
        self.conv_gpb = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)

        out = x_master + x_gpb

        return out


class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        # self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1 = SimpleFPA(C5_size, feature_size)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        # self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        # self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        C3, C4, C5 = inputs

		# remove lateral connection
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        # add activation
        P5_x = self.relu(P5_x)

        P4_x = P5_upsampled_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        # add activation
        P4_x = self.relu(P4_x)

        P3_x = P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        # add activation
        P3_x = self.relu(P3_x)

        return [P3_x, P4_x, P5_x]

class PyramidAttentions(nn.Module):
    def __init__(self, channel_size=256):
        super(PyramidAttentions, self).__init__()

        self.A3_1 = SpatialGate(channel_size)
        self.A3_2 = ChannelGate(channel_size)

        self.A4_1 = SpatialGate(channel_size)
        self.A4_2 = ChannelGate(channel_size)

        self.A5_1 = SpatialGate(channel_size)
        self.A5_2 = ChannelGate(channel_size)

    def forward(self, inputs):
        f3, f4, f5 = inputs

        A3_spatial = self.A3_1(f3)
        A3_channel = self.A3_2(f3)
        A3 = A3_spatial*f3 + A3_channel*f3

        A4_spatial = self.A4_1(f4)
        A4_channel = self.A4_2(f4)
        A4_channel = (A4_channel + A3_channel) / 2
        A4 = A4_spatial*f4 + A4_channel*f4

        A5_spatial = self.A5_1(f5)
        A5_channel = self.A5_2(f5)
        A5_channel = (A5_channel + A4_channel) / 2
        A5 = A5_spatial*f5 + A5_channel*f5

        return [A3, A4, A5, A3_spatial, A4_spatial, A5_spatial]

class SpatialGate(nn.Module):
    """docstring for SpatialGate"""
    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.ConvTranspose2d(out_channels,1,kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)

class ChannelGate(nn.Module):
    """docstring for SpatialGate"""
    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels,out_channels//16,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(out_channels//16,out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = torch.sigmoid(self.conv2(x))
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def generate_anchors_single_pyramid(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack(
        [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (x1, y1, x2, y2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return torch.from_numpy(boxes).cuda()

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.features = features

        self.num_segments = 2

        fpn_sizes = [128, 256, 512, 512]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])
        self.apn = PyramidAttentions()

        self.cls5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

        self.cls4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

        self.cls3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

        self.cls_main = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    # init.xavier_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def get_att_roi(self, att_mask, feature_stride, anchor_size, img_h, img_w, iou_thred=0.2, topk=1):
        with torch.no_grad():
            roi_ret_nms = []
            n, c, h, w = att_mask.size()
            att_corner_unmask = torch.zeros_like(att_mask).cuda()
            if self.num_classes == 200:
                att_corner_unmask[:, :, int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)] = 1
            else:
                att_corner_unmask[:, :, int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)] = 1
            att_mask = att_mask * att_corner_unmask
            feat_anchor = generate_anchors_single_pyramid([anchor_size], [1], [h, w], feature_stride, 1)
            feat_new_cls = att_mask.clone()
            for i in range(n):
                boxes = feat_anchor.clone().float()
                scores = feat_new_cls[i].view(-1)
                score_thred_index = scores > scores.mean()
                boxes = boxes[score_thred_index, :]
                scores = scores[score_thred_index]
                boxes_nms = pth_nms_merge(torch.cat([boxes, scores.unsqueeze(1)], dim=1), iou_thred, topk).cuda()
                boxes_nms[:, 0] = torch.clamp(boxes_nms[:, 0], min=0)
                boxes_nms[:, 1] = torch.clamp(boxes_nms[:, 1], min=0)
                boxes_nms[:, 2] = torch.clamp(boxes_nms[:, 2], max=img_w - 1)
                boxes_nms[:, 3] = torch.clamp(boxes_nms[:, 3], max=img_h - 1)
                roi_ret_nms.append(torch.cat([torch.FloatTensor([i] * boxes_nms.size(0)).unsqueeze(1).cuda(), boxes_nms], 1))

            return torch.cat(roi_ret_nms, 0)

    def get_roi_crop_feat(self, x, roi_list, scale):
        n, c, x2_h, x2_w = x.size()
        roi_3, roi_4, roi_5 = roi_list
        roi_all = torch.cat([roi_3, roi_4, roi_5], 0)
        x2_ret = []
        if self.training:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                roi_3_i = roi_3[roi_3[:, 0] == i] / scale
                roi_4_i = roi_4[roi_4[:, 0] == i] / scale
                # alway drop the roi with highest score
                mask_un = torch.ones(c, x2_h, x2_w).cuda()
                pro_rand = random.random()
                if pro_rand < 0.3:
                    ind_rand = random.randint(0, roi_3_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_3_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_3_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                elif pro_rand < 0.7:
                    ind_rand = random.randint(0, roi_4_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_4_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_4_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                x2_drop = x[i] * mask_un
                x2_crop = x2_drop[:, yy1_resize.long():yy2_resize.long(),
                          xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)

                # normalize
                scale_rate = c*(yy2_resize-yy1_resize)*(xx2_resize-xx1_resize) / torch.sum(mask_un[:, yy1_resize.long():yy2_resize.long(),
                          xx1_resize.long():xx2_resize.long()])
                x2_crop = x2_crop * scale_rate  

                x2_crop_resize = F.upsample(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)
        else:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                x2_crop = x[i, :, yy1_resize.long():yy2_resize.long(),
                          xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                x2_crop_resize = F.upsample(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)
        return torch.cat(x2_ret, 0)


    def forward(self, inputs, targets):
        # inputs.requires_grad = True
        n, c, img_h, img_w = inputs.size()
        
        x3 = checkpoint_sequential(nn.Sequential(*list(self.features.children())[:27]), self.num_segments, inputs)
        x4 = checkpoint_sequential(nn.Sequential(*list(self.features.children())[27:40]), self.num_segments, x3)
        x5 = checkpoint_sequential(nn.Sequential(*list(self.features.children())[40:]), self.num_segments, x4)

        # stage I
        f3, f4, f5 = self.fpn([x3, x4, x5])
        f3_att, f4_att, f5_att, a3, a4, a5 = self.apn([f3, f4, f5])


        out3 = self.cls3(f3_att)
        out4 = self.cls4(f4_att)
        out5 = self.cls5(f5_att)
        loss3 = self.criterion(out3, targets)
        loss4 = self.criterion(out4, targets)
        loss5 = self.criterion(out5, targets)

        # origin classifier
        out_main = self.cls_main(x5)
        loss_main = self.criterion(out_main, targets)

        loss = loss3 + loss4 + loss5 + loss_main
        out = (F.softmax(out3, 1) + F.softmax(out4, 1) + F.softmax(out5, 1) + F.softmax(out_main, 1)) / 4
        _, predicted = torch.max(out.data, 1)
        correct = predicted.eq(targets.data).cpu().sum().item()

        # stage II
        roi_3 = self.get_att_roi(a3, 2 ** 3, 64, img_h, img_w, iou_thred=0.05, topk=5)
        roi_4 = self.get_att_roi(a4, 2 ** 4, 128, img_h, img_w, iou_thred=0.05, topk=3)
        roi_5 = self.get_att_roi(a5, 2 ** 5, 256, img_h, img_w, iou_thred=0.05, topk=1)
        roi_list = [roi_3, roi_4, roi_5]

        x3_crop_resize = self.get_roi_crop_feat(x3, roi_list, 2 ** 3)
        x4_crop_resize = checkpoint_sequential(nn.Sequential(*list(self.features.children())[27:40]), self.num_segments, x3_crop_resize)
        x5_crop_resize = checkpoint_sequential(nn.Sequential(*list(self.features.children())[40:]), self.num_segments, x4_crop_resize)

        f3_crop_resize, f4_crop_resize, f5_crop_resize = self.fpn([x3_crop_resize, x4_crop_resize, x5_crop_resize])
        f3_att_crop_resize, f4_att_crop_resize, f5_att_crop_resize, _, _, _ = self.apn([f3_crop_resize, f4_crop_resize, f5_crop_resize])


        out3_crop_resize = self.cls3(f3_att_crop_resize)
        out4_crop_resize = self.cls4(f4_att_crop_resize)
        out5_crop_resize = self.cls5(f5_att_crop_resize)
        loss3_crop_resize = self.criterion(out3_crop_resize, targets)
        loss4_crop_resize = self.criterion(out4_crop_resize, targets)
        loss5_crop_resize = self.criterion(out5_crop_resize, targets)

        # origin classifier
        out_main_crop_resize = self.cls_main(x5_crop_resize)
        loss_main_crop_resize = self.criterion(out_main_crop_resize, targets)

        loss_crop_resize = loss3_crop_resize + loss4_crop_resize + loss5_crop_resize + loss_main_crop_resize
        out_crop_resize = (F.softmax(out3_crop_resize, 1) + F.softmax(out4_crop_resize, 1) + F.softmax(out5_crop_resize, 1) + F.softmax(out_main_crop_resize, 1)) / 4
        _, predicted_crop_resize = torch.max(out_crop_resize.data, 1)
        correct_crop_resize = predicted_crop_resize.eq(targets.data).cpu().sum().item()


        out_mean = (out_crop_resize + out) / 2
        predicted_mean_, predicted_mean = torch.max(out_mean.data, 1)
        correct_mean = predicted_mean.eq(targets.data).cpu().sum().item()

        loss_ret = {'loss': loss + loss_crop_resize, 'loss1': loss, 'loss2': loss_crop_resize, 'loss3': loss}
        acc_ret = {'acc': correct_mean, 'acc1': correct, 'acc2': correct_crop_resize, 'acc3': correct}

        mask_cat = torch.cat([a3,
                              F.upsample(a4, a3.size()[2:]),
                              F.upsample(a5, a3.size()[2:])], 1)

        return loss_ret, acc_ret, mask_cat, roi_list


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg16(num_class):
    """VGG 16-layer model (configuration "D") with batch normalization

    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_class)

    return model

def vgg19(num_class):

    model = VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_class)

    return model
