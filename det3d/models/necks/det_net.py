import time
from enum import Enum
from functools import reduce

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.models.utils import Empty, GroupNorm, Sequential

import operator
import torch
import warnings

from ..registry import NECKS

@NECKS.register_module
class det_net(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 3, 3, 3],
                 layer_strides=[2, 2, 2, 2],
                 num_filters=[ 64, 128, 256, 512 ],
                 upsample_strides=[1, 2, 4, 4],
                 num_upsample_filters= [ 64, 128, 256, 256, 448 ],
                 num_input_filters=128,
                 encode_background_as_zeros=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 name='det_net'):
        super(det_net, self).__init__()
        assert len(layer_nums) == 4
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        # print('use norm or not')
        # print(use_norm)
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        dimension_feature_map = num_filters #[ 64, 128, 256, 512] # [256, 512, 512, 512]
        dimension_concate     = num_upsample_filters # last one for final output

        # ===============================================================
        # block0
        # ==============================================================
        flag = 0
        middle_layers = []
        for i in range(layer_nums[flag]):
            middle_layers.append( Conv2d(dimension_feature_map[flag], dimension_feature_map[flag], 3, padding=1))
            middle_layers.append( BatchNorm2d(dimension_feature_map[flag]) )
            middle_layers.append( nn.ReLU() )
        self.block0 = Sequential(*middle_layers)
        middle_layers = []
        self.downsample0 = Sequential(
            Conv2d(dimension_feature_map[flag], dimension_concate[flag], 3, stride = 2 ),
            BatchNorm2d(dimension_concate[flag]),
            nn.ReLU(),
        )


        # ===============================================================
        # block1
        # ==============================================================
        flag = 1
        middle_layers = []
        for i in range(layer_nums[flag]):
            middle_layers.append( Conv2d(dimension_feature_map[flag], dimension_feature_map[flag], 3, padding=1))
            middle_layers.append( BatchNorm2d(dimension_feature_map[flag]) )
            middle_layers.append( nn.ReLU() )
        self.block1 = Sequential(*middle_layers)



        # ===============================================================
        # block2
        # ==============================================================
        flag = 2
        middle_layers = []
        for i in range(layer_nums[flag]):
            middle_layers.append( Conv2d(dimension_feature_map[flag], dimension_feature_map[flag], 3, padding=1))
            middle_layers.append( BatchNorm2d(dimension_feature_map[flag]) )
            middle_layers.append( nn.ReLU() )
        self.block2 = Sequential(*middle_layers)


        # ===============================================================
        # block3
        # ==============================================================
        flag = 3
        middle_layers = []
        for i in range(layer_nums[flag]):
            middle_layers.append( Conv2d(dimension_feature_map[flag], dimension_feature_map[flag], 3, padding=1))
            middle_layers.append( BatchNorm2d(dimension_feature_map[flag]) )
            middle_layers.append( nn.ReLU() )
        self.block3 = Sequential(*middle_layers)
        self.upsample3 = Sequential(
            ConvTranspose2d(dimension_feature_map[flag], dimension_concate[flag], 3, stride = 2 ),
            BatchNorm2d( dimension_concate[flag] ),
            nn.ReLU(),
        )

        # ==============================================================
        # convlution after concatating block3 and block2
        # ==============================================================
        middle_layers = []
        middle_layers.append( Conv2d( (dimension_concate[3]+dimension_feature_map[2]), dimension_concate[2], 3, padding=1))
        middle_layers.append( BatchNorm2d(dimension_concate[2]) )
        middle_layers.append( nn.ReLU() )
        middle_layers.append( Conv2d(dimension_concate[2] , dimension_concate[2], 3, padding=1))
        middle_layers.append( BatchNorm2d(dimension_concate[2]) )
        middle_layers.append( nn.ReLU() )
        # upsampling
        middle_layers.append( ConvTranspose2d( dimension_concate[2] , dimension_concate[2], 3, stride = 2))
        middle_layers.append( BatchNorm2d(dimension_concate[2]) )
        middle_layers.append( nn.ReLU() )

        self.upsample2_after_concate_fuse32 = Sequential(*middle_layers)

        # ==============================================================
        # convlution after concatating block2, block1 and block0
        # ==============================================================
        middle_layers = []
        middle_layers.append( Conv2d( ( dimension_concate[0] + dimension_feature_map[1] + dimension_concate[2]), dimension_concate[4], 3, padding=1))
        middle_layers.append( BatchNorm2d(dimension_concate[4]) )
        middle_layers.append( nn.ReLU() )
        middle_layers.append( Conv2d(dimension_concate[4] , dimension_concate[4], 3, padding=1))
        middle_layers.append( BatchNorm2d( dimension_concate[4] ))
        middle_layers.append( nn.ReLU() )
        self.output_after_concate_fuse210 = Sequential(*middle_layers)


    def forward(self, x, bev=None):

        down0= self.downsample0(self.block0(x[0]))
        x1   = self.block1(x[1])
        x2   = self.block2(x[2])
        up3  = self.upsample3(self.block3(x[3]))
        # concate32
        fuse32 = torch.cat([x2, up3], dim=1)
        fuse32 = self.upsample2_after_concate_fuse32(fuse32)
        # concate210
        fuse210= torch.cat([down0, x1, fuse32 ],dim=1)
        xx = self.output_after_concate_fuse210(fuse210)

        return xx


@NECKS.register_module
class det_net_2(nn.Module):
    '''
    directly upsample without smoothly fusing
    '''
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 3, 3, 3],
                 layer_strides=[2, 2, 2, 2],
                 num_filters=[128, 128, 128, 128],
                 upsample_strides=[1, 2, 4, 4],
                 num_upsample_filters= [128, 128, 128, 128, 256],# [256, 128, 256, 256, 256],
                 num_input_filters=128,
                 encode_background_as_zeros=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 name='det_net_2'):
        super(det_net_2, self).__init__()
        assert len(layer_nums) == 4
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        dimension_feature_map = num_filters #[ 64, 128, 256, 512] # [256, 512, 512, 512]
        dimension_concate     = num_upsample_filters # last one for final output

        # ===============================================================
        # block0
        # ==============================================================
        flag = 0
        middle_layers = []
        for i in range(layer_nums[flag]):
            middle_layers.append( Conv2d(dimension_feature_map[flag], dimension_feature_map[flag], 3, padding=1))
            middle_layers.append( BatchNorm2d(dimension_feature_map[flag]) )
            middle_layers.append( nn.ReLU() )
        self.block0 = Sequential(*middle_layers)
        middle_layers = []
        self.downsample0 = Sequential(
            Conv2d(dimension_feature_map[flag], dimension_concate[flag], 3, stride = 2 ),
            BatchNorm2d(dimension_concate[flag]),
            nn.ReLU(),
        )


        # ===============================================================
        # block1
        # ==============================================================
        flag = 1
        middle_layers = []
        for i in range(layer_nums[flag]):
            middle_layers.append( Conv2d(dimension_feature_map[flag], dimension_feature_map[flag], 3, padding=1))
            middle_layers.append( BatchNorm2d(dimension_feature_map[flag]) )
            middle_layers.append( nn.ReLU() )
        self.block1 = Sequential(*middle_layers)



        # ===============================================================
        # block2
        # ==============================================================
        flag = 2
        middle_layers = []
        for i in range(layer_nums[flag]):
            middle_layers.append( Conv2d(dimension_feature_map[flag], dimension_feature_map[flag], 3, padding=1))
            middle_layers.append( BatchNorm2d(dimension_feature_map[flag]) )
            middle_layers.append( nn.ReLU() )
        self.block2 = Sequential(*middle_layers)

        self.upsample2 = Sequential(
            ConvTranspose2d(dimension_feature_map[flag], dimension_concate[flag], 3, stride = 2),
            BatchNorm2d( dimension_concate[flag] ),
            nn.ReLU(),
        )


        # ===============================================================
        # block3
        # ==============================================================
        flag = 3
        middle_layers = []
        for i in range(layer_nums[flag]):
            middle_layers.append( Conv2d(dimension_feature_map[flag], dimension_feature_map[flag], 3, padding=1))
            middle_layers.append( BatchNorm2d(dimension_feature_map[flag]) )
            middle_layers.append( nn.ReLU() )
        self.block3 = Sequential(*middle_layers)

        self.upsample3 = Sequential(
            ConvTranspose2d(dimension_feature_map[flag], dimension_concate[flag], 5, stride = 4, output_padding = 2 ), # (49-1)*4 +5 +2
            BatchNorm2d( dimension_concate[flag] ),
            nn.ReLU(),
        )

        # ==============================================================
        # output layer
        # ==============================================================
        middle_layers = []
        middle_layers.append( Conv2d( ( dimension_concate[0] + dimension_feature_map[1] + dimension_concate[2] + dimension_concate[3]), dimension_concate[4], 3, padding=1))
        middle_layers.append( BatchNorm2d(dimension_concate[4]) )
        middle_layers.append( nn.ReLU() )
        middle_layers.append( Conv2d(dimension_concate[4] , dimension_concate[4], 3, padding=1))
        middle_layers.append( BatchNorm2d( dimension_concate[4] ))
        middle_layers.append( nn.ReLU() )
        self.output_after_concate_fuse3210 = Sequential(*middle_layers)

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")


    def forward(self, x, bev=None):

        x0 = self.downsample0(self.block0(x[0]))
        x1 = self.block1(x[1])
        x2 = self.upsample2(self.block2(x[2]))
        x3 = self.upsample3(self.block3(x[3]))

        xx = self.output_after_concate_fuse3210( torch.cat([x0, x1, x2, x3 ],dim=1) )

        return xx



