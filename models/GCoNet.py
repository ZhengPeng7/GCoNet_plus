from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import fvcore.nn.weight_init as weight_init
from torchvision.models import resnet50

from models.modules import ResBlk, DSLayer, half_DSLayer, CoAttLayer

from config import Config


class GCoNet(nn.Module):
    def __init__(self, bb=Config().bb):
        super(GCoNet, self).__init__()
        if bb == 'vgg16':
            bb_vgg16 = list(vgg16(pretrained=True).children())[0]
            bb_convs = OrderedDict({
                'conv1': bb_vgg16[:4],
                'conv2': bb_vgg16[4:9],
                'conv3': bb_vgg16[9:16],
                'conv4': bb_vgg16[16:23],
                'conv5': bb_vgg16[23:30]
            })
            channel_scale = 1
        elif bb == 'resnet50':
            bb_resnet50 = list(resnet50(pretrained=True).children())
            bb_convs = OrderedDict({
                'conv1': nn.Sequential(*bb_resnet50[0:3]),
                'conv2': bb_resnet50[4],
                'conv3': bb_resnet50[5],
                'conv4': bb_resnet50[6],
                'conv5': bb_resnet50[7]
            })
            channel_scale = 4
        self.bb = nn.Sequential(bb_convs)

        self.top_layer = nn.Sequential(
            nn.Conv2d(512*channel_scale, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        )

        channel_scale_latlayer = channel_scale // 2 if bb == 'resnet50' else 1
        self.latlayer4 = ResBlk(channel_in=512*channel_scale_latlayer)
        self.latlayer3 = ResBlk(channel_in=256*channel_scale_latlayer)
        self.latlayer2 = ResBlk(channel_in=128*channel_scale_latlayer)
        self.latlayer1 = ResBlk(channel_in=64*1)

        self.enlayer4 = ResBlk()
        self.enlayer3 = ResBlk()
        self.enlayer2 = ResBlk()
        self.enlayer1 = ResBlk()

        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()

        self.pred_layer = half_DSLayer(512*channel_scale)

        self.co_x5 = CoAttLayer(channel_in=512*channel_scale)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*channel_scale, 291)       # DUTS_class has 291 classes

        for layer in [self.classifier]:
            weight_init.c2_msra_fill(layer)

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def _fg_att(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
        return feat * pred

    def forward(self, x):
        [_, _, H, W] = x.size()
        x1 = self.bb.conv1(x)
        x2 = self.bb.conv2(x1)
        x3 = self.bb.conv3(x2)
        x4 = self.bb.conv4(x3)
        x5 = self.bb.conv5(x4)

        _x5 = self.avgpool(x5)
        _x5 = _x5.view(_x5.size(0), -1)
        pred_cls = self.classifier(_x5)

        weighted_x5, neg_x5 = self.co_x5(x5)

        cam = torch.mean(weighted_x5, dim=1).unsqueeze(1)
        cam = cam.sigmoid()
        if self.training:
            ########## contrastive branch #########
            cat_x5 = torch.cat([weighted_x5, neg_x5], dim=0)
            pred_x5 = self.pred_layer(cat_x5)
            pred_x5 = F.interpolate(pred_x5, size=(H, W), mode='bilinear', align_corners=True)

        ########## Up-Sample ##########
        preds = []
        p5 = self.top_layer(weighted_x5)
        _pred = cam
        preds.append(F.interpolate(_pred, size=(H, W), mode='bilinear', align_corners=True))

        p4 = self._upsample_add(p5, self.latlayer4(x4)) 
        p4 = self.enlayer4(p4)
        _pred = self.dslayer4(p4)
        preds.append(F.interpolate(_pred, size=(H, W), mode='bilinear', align_corners=True))

        p3 = self._upsample_add(p4, self.latlayer3(x3)) 
        p3 = self.enlayer3(p3)
        _pred = self.dslayer3(p3)
        preds.append(F.interpolate(_pred, size=(H, W), mode='bilinear', align_corners=True))

        p2 = self._upsample_add(p3, self.latlayer2(x2)) 
        p2 = self.enlayer2(p2)
        _pred = self.dslayer2(p2)
        preds.append(F.interpolate(_pred, size=(H, W), mode='bilinear', align_corners=True))

        p1 = self._upsample_add(p2, self.latlayer1(x1)) 
        p1 = self.enlayer1(p1)
        _pred = self.dslayer1(p1)
        preds.append(F.interpolate(_pred, size=(H, W), mode='bilinear', align_corners=True))

        if self.training:
            return preds, pred_cls, pred_x5
        else:
            return preds

