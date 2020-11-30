# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

# high_resoluton_net related params for segmentation
HRNet = CN()
HRNet.PRETRAINED_LAYERS = ['*']
HRNet.STEM_INPLANES = 64
HRNet.FINAL_CONV_KERNEL = 1
HRNet.WITH_HEAD = True
HRNet.NUM_CLASSES = 19
HRNet.PRETRAINED = ''

HRNet.STAGE1 = CN()
HRNet.STAGE1.NUM_MODULES = 1
HRNet.STAGE1.NUM_BRANCHES = 1
HRNet.STAGE1.NUM_BLOCKS = [4]
HRNet.STAGE1.NUM_CHANNELS = [32]
HRNet.STAGE1.BLOCK = 'BOTTLENECK'
HRNet.STAGE1.FUSE_METHOD = 'SUM'

HRNet.STAGE2 = CN()
HRNet.STAGE2.NUM_MODULES = 1
HRNet.STAGE2.NUM_BRANCHES = 2
HRNet.STAGE2.NUM_BLOCKS = [4, 4]
HRNet.STAGE2.NUM_CHANNELS = [32, 64]
HRNet.STAGE2.BLOCK = 'BASIC'
HRNet.STAGE2.FUSE_METHOD = 'SUM'

HRNet.STAGE3 = CN()
HRNet.STAGE3.NUM_MODULES = 1
HRNet.STAGE3.NUM_BRANCHES = 3
HRNet.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNet.STAGE3.NUM_CHANNELS = [32, 64, 128]
HRNet.STAGE3.BLOCK = 'BASIC'
HRNet.STAGE3.FUSE_METHOD = 'SUM'

HRNet.STAGE4 = CN()
HRNet.STAGE4.NUM_MODULES = 1
HRNet.STAGE4.NUM_BRANCHES = 4
HRNet.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNet.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HRNet.STAGE4.BLOCK = 'BASIC'
HRNet.STAGE4.FUSE_METHOD = 'SUM'

