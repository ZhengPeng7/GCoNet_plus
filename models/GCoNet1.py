import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision.models import vgg16
import fvcore.nn.weight_init as weight_init

class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x

class half_DSLayer(nn.Module):
    def __init__(self, in_channel=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(int(in_channel/2), int(in_channel/4), kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel/4), 1, kernel_size=1, stride=1, padding=0)) #, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class AllAttLayer(nn.Module):
    def __init__(self, channel_in=512):

        super(AllAttLayer, self).__init__()
        self.query_transform = Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.key_transform = Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (channel_in ** 0.5)

        self.conv6 = Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        for layer in [self.query_transform, self.key_transform, self.conv6]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x5):
        # x: B,C,H,W
        # x_query: B,C,HW
        B, C, H5, W5 = x5.size()

        x_query = self.query_transform(x5).view(B, C, -1)

        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2).contiguous().view(-1, C) # BHW, C
        # x_key: B,C,HW
        x_key = self.key_transform(x5).view(B, C, -1)

        x_key = torch.transpose(x_key, 0, 1).contiguous().view(C, -1) # C, BHW

        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key) #* self.scale # BHW, BHW
        x_w = x_w.view(B*H5*W5, B, H5*W5)
        x_w = torch.max(x_w, -1).values # BHW, B
        x_w = x_w.mean(-1)
        #x_w = torch.mean(x_w, -1).values # BHW
        x_w = x_w.view(B, -1) * self.scale # B, HW
        x_w = F.softmax(x_w, dim=-1) # B, HW
        x_w = x_w.view(B, H5, W5).unsqueeze(1) # B, 1, H, W
 
        x5 = x5 * x_w
        x5 = self.conv6(x5)

        return x5

class CoAttLayer(nn.Module):
    def __init__(self, channel_in=512):

        super(CoAttLayer, self).__init__()

        self.all_attention = AllAttLayer(channel_in)
        self.conv_output = Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.conv_transform = Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.fc_transform = nn.Linear(channel_in, channel_in)

        for layer in [self.conv_output, self.conv_transform, self.fc_transform]:
            weight_init.c2_msra_fill(layer)
    
    def forward(self, x5):
        if self.training:
            f_begin = 0
            f_end = int(x5.shape[0] / 2)
            s_begin = f_end
            s_end = int(x5.shape[0])

            x5_1 = x5[f_begin: f_end]
            x5_2 = x5[s_begin: s_end]

            x5_new_1 = self.all_attention(x5_1)
            x5_new_2 = self.all_attention(x5_2)

            x5_1_proto = torch.mean(x5_new_1, (0, 2, 3), True).view(1, -1)
            x5_1_proto = x5_1_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            x5_2_proto = torch.mean(x5_new_2, (0, 2, 3), True).view(1, -1)
            x5_2_proto = x5_2_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            x5_11 = x5_1 * x5_1_proto
            x5_22 = x5_2 * x5_2_proto
            weighted_x5 = torch.cat([x5_11, x5_22], dim=0)

            x5_12 = x5_1 * x5_2_proto
            x5_21 = x5_2 * x5_1_proto
            neg_x5 = torch.cat([x5_12, x5_21], dim=0)
        else:

            x5_new = self.all_attention(x5)
            x5_proto = torch.mean(x5_new, (0, 2, 3), True).view(1, -1)
            x5_proto = x5_proto.unsqueeze(-1).unsqueeze(-1) # 1, C, 1, 1

            weighted_x5 = x5 * x5_proto #* cweight
            neg_x5 = None
        return weighted_x5, neg_x5


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class GINet(nn.Module):
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, mode='train'):
        super(GINet, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.mode = mode

        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))

        self.latlayer4 = LatLayer(in_channel=512)
        self.latlayer3 = LatLayer(in_channel=256)
        self.latlayer2 = LatLayer(in_channel=128)
        self.latlayer1 = LatLayer(in_channel=64)

        self.enlayer4 = EnLayer()
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()

        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()

        self.pred_layer = half_DSLayer(512)

        self.co_x5 = CoAttLayer()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 291)

        for layer in [self.classifier]:
            weight_init.c2_msra_fill(layer)

    def set_mode(self, mode):
        self.mode = mode

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        if self.mode == 'train':
            preds = self._forward(x)
        else:
            with torch.no_grad():
                preds = self._forward(x)

        return preds

    def _forward(self, x):
        [_, _, H, W] = x.size()
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)
        x5 = self.backbone.conv5(x4)

        _x5 = self.avgpool(x5)
        _x5 = _x5.view(_x5.size(0), -1)
        pred_cls = self.classifier(_x5)

        weighted_x5, neg_x5 = self.co_x5(x5)

        if self.training:
            ########## contrastive branch #########
            cat_x5 = torch.cat([weighted_x5, neg_x5], dim=0)
            pred_x5 = self.pred_layer(cat_x5)
            pred_x5 = F.interpolate(pred_x5,
                              size=(H, W),
                              mode='bilinear',
                              align_corners=True)

        ########## Up-Sample ##########
        preds = []
        p5 = self.toplayer(weighted_x5)

        p4 = self._upsample_add(p5, self.latlayer4(x4)) 
        p4 = self.enlayer4(p4)
        _pred = self.dslayer4(p4)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p3 = self._upsample_add(p4, self.latlayer3(x3)) 
        p3 = self.enlayer3(p3)
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p2 = self._upsample_add(p3, self.latlayer2(x2)) 
        p2 = self.enlayer2(p2)
        _pred = self.dslayer2(p2)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p1 = self._upsample_add(p2, self.latlayer1(x1)) 
        p1 = self.enlayer1(p1)
        _pred = self.dslayer1(p1)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        if self.training:
            return preds, pred_cls, pred_x5
        else:
            return preds


class GCoNet(nn.Module):
    def __init__(self, mode='train'):
        super(GCoNet, self).__init__()
        self.co_classifier = vgg16(pretrained=True).eval()
        self.ginet = GINet()
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode
        self.ginet.set_mode(self.mode)

    def forward(self, x):
        ########## Co-SOD ############
        preds = self.ginet(x)

        return preds


class VGG_Backbone(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self):
        super(VGG_Backbone, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.conv1 = conv1

        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3', nn.MaxPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4.add_module('relu4_1', nn.ReLU())
        conv4.add_module('conv4_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_2', nn.ReLU())
        conv4.add_module('conv4_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_3', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4', nn.MaxPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_1', nn.ReLU())
        conv5.add_module('conv5_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_2', nn.ReLU())
        conv5.add_module('conv5_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_3', nn.ReLU())
        self.conv5 = conv5

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )
        
        pre_train = torch.load(os.path.join('/home/pz1/.cache/torch/hub/checkpoints', 'vgg16-397923af.pth'))
        self._initialize_weights(pre_train)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4_1(x)
        x1 = self.conv5_1(x1)
        x1 = self.avgpool(x1)
        _x1 = x1.view(x1.size(0), -1)
        pred_vector = self.classifier(_x1)

        x2 = self.conv4_2(x)
        x2 = self.conv5_2(x2)
        return x1, pred_vector, x2

    def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())
        self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])
        self.conv4.conv4_1.weight.data.copy_(pre_train[keys[14]])
        self.conv4.conv4_2.weight.data.copy_(pre_train[keys[16]])
        self.conv4.conv4_3.weight.data.copy_(pre_train[keys[18]])
        self.conv5.conv5_1.weight.data.copy_(pre_train[keys[20]])
        self.conv5.conv5_2.weight.data.copy_(pre_train[keys[22]])
        self.conv5.conv5_3.weight.data.copy_(pre_train[keys[24]])

        self.conv1.conv1_1.bias.data.copy_(pre_train[keys[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])
        self.conv4.conv4_1.bias.data.copy_(pre_train[keys[15]])
        self.conv4.conv4_2.bias.data.copy_(pre_train[keys[17]])
        self.conv4.conv4_3.bias.data.copy_(pre_train[keys[19]])
        self.conv5.conv5_1.bias.data.copy_(pre_train[keys[21]])
        self.conv5.conv5_2.bias.data.copy_(pre_train[keys[23]])
        self.conv5.conv5_3.bias.data.copy_(pre_train[keys[25]])

        self.classifier[0].weight.data.copy_(pre_train[keys[26]])
        self.classifier[0].bias.data.copy_(pre_train[keys[27]])
        self.classifier[3].weight.data.copy_(pre_train[keys[28]])
        self.classifier[3].bias.data.copy_(pre_train[keys[29]])
        self.classifier[6].weight.data.copy_(pre_train[keys[30]])
        self.classifier[6].bias.data.copy_(pre_train[keys[31]])
