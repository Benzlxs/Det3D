import time

import numpy as np
import spconv
import torch
from det3d.models.utils import Empty, change_default_args
from det3d.torchie.cnn import constant_init, kaiming_init
from det3d.torchie.trainer import load_checkpoint
from spconv import SparseConv3d, SubMConv3d, SubMConv2d, SparseInverseConv3d, JoinTable, Identity, ConcatTable
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .. import builder
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        _identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            _identity = self.downsample(x)

        out.features += _identity.features
        out.features = self.relu(out.features)

        return out

@BACKBONES.register_module
class tDBN_1(nn.Module):
    def __init__(
            self, num_input_features=128, norm_cfg=None, name="tDBN_1", **kwargs
            ):
        super(tDBN_1, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        middle_layers = spconv.SparseSequential()

        num_filter_fpn = [ 32, 64, 96, 128 ]
        dimension_feature_map = [ 128, 128, 128, 128 ]
        dimension_kernel_size = [15,  7,   3,  1 ]

        for i, o in [[num_input_features, num_filter_fpn[0]]]:
            middle_layers.add(SubMConv3d(i, o, 3, indice_key="subm_0", bias = False))
            middle_layers.add(build_norm_layer(norm_cfg, o)[1])
            middle_layers.add(nn.ReLU())

        self.block0 = middle_layers
        middle_layers = spconv.SparseSequential()

        middle_layers.add(
            SparseConv3d(
                num_filter_fpn[0],
                dimension_feature_map[0],
                (dimension_kernel_size[0], 1, 1), (2, 1, 1),bias=False)
            )
        middle_layers.add(build_norm_layer(norm_cfg, dimension_feature_map[0])[1])
        middle_layers.add(nn.ReLU())
        # dense function add later
        self.feature_map0 =  middle_layers

        middle_layers = spconv.SparseSequential()
         ## block1-3 and feature map1-3
        for k in range(1, 4):
            middle_layers.add(
                SparseConv3d(
                    num_filter_fpn[k-1],
                    num_filter_fpn[k],
                    3, 2, bias=False)
                    )
            middle_layers.add(build_norm_layer(norm_cfg, num_filter_fpn[k])[1])
            middle_layers.add(nn.ReLU())
            # dense function add later

            # 128*7*199*175 recurrent
            for i, o in [[num_filter_fpn[k], num_filter_fpn[k]], [num_filter_fpn[k], num_filter_fpn[k]]]:
                middle_layers.add(SubMConv3d(i, o, 3, indice_key="subm_{}".format(k) , bias = False))
                middle_layers.add(build_norm_layer(norm_cfg, o)[1])
                middle_layers.add(nn.ReLU())

            if k==1:
                self.block1 = middle_layers
            elif k==2:
                self.block2 = middle_layers
            elif k==3:
                self.block3 = middle_layers

            middle_layers = spconv.SparseSequential()

            middle_layers.add(
                SparseConv3d(
                    num_filter_fpn[k],
                    dimension_feature_map[k],
                    (dimension_kernel_size[k], 1, 1), (2, 1, 1),bias=False)
                )
            middle_layers.add(build_norm_layer(norm_cfg, dimension_feature_map[k])[1])
            middle_layers.add(nn.ReLU())
            # adding dense in forward step

            if   k==1 :
                self.feature_map1 = middle_layers
            elif k==2:
                self.feature_map2 = middle_layers
            elif k==3:
                self.feature_map3 = None   # convert the sparse data into dense one
                # Sequential(scn.SparseToDense(3, dimension_feature_map[k]))  ## last one is the 2D instead of 3D

            middle_layers = spconv.SparseSequential()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")


    def forward(self, voxel_features, coors, batch_size, input_shape):
        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        output = {}
        for k in range(4):
            if k == 0:
                ret = self.block0(ret)
            elif k==1:
                ret = self.block1(ret)
            elif k==2:
                ret = self.block2(ret)
            elif k==3:
                ret = self.block3(ret)

            temp = []

            if k==0:
                temp = self.feature_map0(ret).dense() # D: 5
            elif k==1:
                temp = self.feature_map1(ret).dense() # D: 3
            elif k==2:
                temp = self.feature_map2(ret).dense() # D: 2
            elif k==3:
                temp = ret.dense()

            N, C, D, H, W = temp.shape
            output[k] = temp.view(N, C*D, H, W)

        return output


@BACKBONES.register_module
class tDBN_2(nn.Module):
    def __init__(
            self, num_input_features=128, norm_cfg=None, name="tDBN_2", **kwargs
            ):
        super(tDBN_2, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        middle_layers = spconv.SparseSequential()

        num_filter_fpn = [ 32, 64, 96, 128 ]
        dimension_feature_map = [ 128, 128, 128, 128 ]
        dimension_kernel_size = [15,  7,   3,  1 ]

        # the main network framework
        m = spconv.SparseSequential()
        # --------------------------------------
        ## block1 and feature map 0, convert from voxel into 3D tensor
        # --------------------------------------
        for i, o in [[num_input_features, num_filter_fpn[0]]]: # , [num_filter_fpn[0], num_filter_fpn[0]]]:
            middle_layers.add(SubMConv3d(i, o, 3, indice_key="subm_0", bias = False))
            middle_layers.add(build_norm_layer(norm_cfg, o)[1])
            middle_layers.add(nn.ReLU())

        self.block_input = m
        m = spconv.SparseSequential()

        reps = 2
        dimension = 3
        residual_use = False # using residual block or not
        _index = 0
        for _ in range(reps):
            self.block(m, num_filter_fpn[0], num_filter_fpn[0], index=_index, residual_blocks = residual_use)

        self.x0_in = m

        for k in range(1,4):
            m = spconv.SparseSequential()
            m.add(
                SparseConv3d( num_filter_fpn[k-1], num_filter_fpn[k], 3, 2, bias=False)
                )
            m.add(build_norm_layer(norm_cfg, num_filter_fpn[k])[1])
            m.add(nn.ReLU())

            for _ in range(reps):
                if k==4:
                    self.block(m, num_filter_fpn[k], num_filter_fpn[k], dimension=2, residual_blocks = residual_use)
                else:
                    self.block(m, num_filter_fpn[k], num_filter_fpn[k], dimension=3, residual_blocks = residual_use)
            if k==1:
                self.x1_in = m
            elif k==2:
                self.x2_in = m
            elif k==3:
                self.x3_in = m

        for k in range(2, -1, -1):
            m = spconv.SparseSequential()
            m.add(
                SparseInverseConv3d(num_filter_fpn[k+1], num_filter_fpn[k], 3, 2, bias=False)
                    )
            m.add(build_norm_layer(norm_cfg, num_filter_fpn[k])[1])
            m.add(nn.ReLU())

            if k==2:
                self.upsample32 = m
            elif k==1:
                self.upsample21 = m
            elif k==0:
                self.upsample10 = m

            m = spconv.SparseSequential()
            m.add(JoinTable())

            for i in range(reps):
                self.block(m, num_filter_fpn[k] * (2 if i == 0 else 1), num_filter_fpn[k], residual_blocks = residual_use)

            if k==2:
                self.concate2 = m
            elif k==1:
                self.concate1 = m
            elif k==0:
                self.concate0 = m

            m = spconv.SparseSequential()

            m.add(
                SparseConv3d(num_filter_fpn[k], dimension_feature_map[k], (dimension_kernel_size[k], 1, 1), (1, 1, 1),bias=False)
                    )
            m.add(build_norm_layer(norm_cfg, dimension_feature_map[k])[1])
            m.add(nn.ReLU())


            if k==2:
                self.feature_map2 = m
            elif k==1:
                self.feature_map1 = m
            elif k==0:
                self.feature_map0 = m





    def block(self, m, i, o, dimension=3, index=0, residual_blocks=False):  # default using residual_block
        if dimension == 3:  ## 3x3x3 convlution
            if residual_blocks: #ResNet style blocks
                m.add(ConcatTable().add(
                    Identity()).add(
                        spconv.SparseSequential().add(
                            SubMConv3d(i, o, 3, indice_key="su3_{}".format(index), bias = False)).add(build_norm_layer(norm_cfg, o)[1]).add(nn.ReLU()).add(
                            SubMConv3d(i, o, 3, indice_key="su3_{}".format(index), bias = False)).add(build_norm_layer(norm_cfg, o)[1]).add(nn.ReLU())
                                        )
                    ).add(JoinTable())
            else:
                m.add(
                    spconv.SparseSequential().add(
                        SubMConv3d(i, o, 3, indice_key="su3_{}".format(index), bias = False)).add(build_norm_layer(norm_cfg, o)[1]).add(nn.ReLU()).add(
                        SubMConv3d(i, o, 3, indice_key="su3_{}".format(index), bias = False)).add(build_norm_layer(norm_cfg, o)[1]).add(nn.ReLU())
                        )
        else:   # 2x2x2 convoltion
            if residual_blocks: #ResNet style blocks
                m.add(ConcatTable().add(
                    Identity()).add(
                        spconv.SparseSequential().add(
                            SubMConv2d(i, o, 3, indice_key="su2_{}".format(index), bias = False)).add(build_norm_layer(norm_cfg, o)[1]).add(nn.ReLU()).add(
                            SubMConv2d(i, o, 3, indice_key="su2_{}".format(index), bias = False)).add(build_norm_layer(norm_cfg, o)[1]).add(nn.ReLU())
                                        )
                    ).add(JoinTable())
            else:
                m.add(
                    spconv.SparseSequential().add(
                        SubMConv2d(i, o, 3, indice_key="su2_{}".format(index), bias = False)).add(build_norm_layer(norm_cfg, o)[1]).add(nn.ReLU()).add(
                        SubMConv2d(i, o, 3, indice_key="su2_{}".format(index), bias = False)).add(build_norm_layer(norm_cfg, o)[1]).add(nn.ReLU())
                        )




    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")


    def forward(self, voxel_features, coors, batch_size, input_shape):
        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        output = {}
        x0 = self.x0_in(ret)
        x1 = self.x1_in(x0)
        x2 = self.x2_in(x1)
        x3 = self.x3_in(x2)

        x2_f = self.concate2([x2, self.upsample32(x3)])
        x1_f = self.concate1([x1, self.upsample21(x2_f)])
        x0_f = self.concate0([x0, self.upsample10(x1_f)])

        # generate output feature maps
        x0_out = self.feature_map0(x0_f).dense()
        x1_out = self.feature_map1(x1_f).dense()
        x2_out = self.feature_map2(x2_f).dense()
        x3_out = self.feature_map3(x3).dense()

        # output
        N, C, D, H, W = x0_out.shape
        output[0] = x0_out.view(N, C*D, H, W)

        N, C, D, H, W = x1_out.shape
        output[1] = x1_out.view(N, C*D, H, W)

        N, C, D, H, W = x2_out.shape
        output[2] = x2_out.view(N, C*D, H, W)

        N, C, D, H, W = x3_out.shape
        output[3] = x3_out.view(N, C*D, H, W)

        return output


