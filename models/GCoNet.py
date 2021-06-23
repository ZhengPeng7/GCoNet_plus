from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import fvcore.nn.weight_init as weight_init
from torchvision.models import resnet50

from models.modules import ResBlk, DSLayer, half_DSLayer, CoAttLayer

from config import Config


config = Config()


class GCoNet(nn.Module):
    def __init__(self, bb=config.bb):
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

        channel_last = 32
        activation_out = config.activation_out
        if config.loss_cls_mask_last_layers == 1:
            self.dslayer4 = DSLayer(channel_out=1, activation_out=nn.Sigmoid())
            self.dslayer3 = DSLayer(channel_out=1, activation_out=nn.Sigmoid())
            self.dslayer2 = DSLayer(channel_out=1, activation_out=nn.Sigmoid())
        elif config.loss_cls_mask_last_layers == 2:
            self.dslayer4 = DSLayer(channel_out=1, activation_out=nn.Sigmoid())
            self.dslayer3 = DSLayer(channel_out=1, activation_out=nn.Sigmoid())
            self.dslayer2 = DSLayer(channel_out=channel_last, activation_out=(nn.ReLU(inplace=True) if activation_out == 'relu' else nn.Sigmoid()))
        elif config.loss_cls_mask_last_layers == 3:
            self.dslayer4 = DSLayer(channel_out=1, activation_out=nn.Sigmoid())
            self.dslayer3 = DSLayer(channel_out=channel_last, activation_out=(nn.ReLU(inplace=True) if activation_out == 'relu' else nn.Sigmoid()))
            self.dslayer2 = DSLayer(channel_out=channel_last, activation_out=(nn.ReLU(inplace=True) if activation_out == 'relu' else nn.Sigmoid()))
        elif config.loss_cls_mask_last_layers == 4:
            self.dslayer4 = DSLayer(channel_out=channel_last, activation_out=(nn.ReLU(inplace=True) if activation_out == 'relu' else nn.Sigmoid()))
            self.dslayer3 = DSLayer(channel_out=channel_last, activation_out=(nn.ReLU(inplace=True) if activation_out == 'relu' else nn.Sigmoid()))
            self.dslayer2 = DSLayer(channel_out=channel_last, activation_out=(nn.ReLU(inplace=True) if activation_out == 'relu' else nn.Sigmoid()))
        self.dslayer1 = DSLayer(channel_out=channel_last, activation_out=(nn.ReLU(inplace=True) if activation_out == 'relu' else nn.Sigmoid()))

        if config.GAM:
            self.co_x5 = CoAttLayer(channel_in=512*channel_scale)

        if 'contrast' in config.loss:
            self.pred_layer = half_DSLayer(512*channel_scale)

        if {'cls', 'cls_mask'} & set(config.loss):
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(512*channel_scale, 291)       # DUTS_class has 291 classes
            for layer in [self.classifier]:
                weight_init.c2_msra_fill(layer)

        self.convs_out = []
        for _ in range(config.loss_cls_mask_last_layers):
            self.convs_out.append(nn.Sequential(nn.Conv2d(channel_last, 1, 1, 1, 0).cuda(), nn.Sigmoid().cuda()))

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        [N, _, H, W] = x.size()
        x1 = self.bb.conv1(x)
        x2 = self.bb.conv2(x1)
        x3 = self.bb.conv3(x2)
        x4 = self.bb.conv4(x3)
        x5 = self.bb.conv5(x4)

        if 'cls' in config.loss:
            _x5 = self.avgpool(x5)
            _x5 = _x5.view(_x5.size(0), -1)
            pred_cls = self.classifier(_x5)

        if config.GAM:
            weighted_x5, neg_x5 = self.co_x5(x5)
            if 'contrast' in config.loss:
                if self.training:
                    ########## contrastive branch #########
                    cat_x5 = torch.cat([weighted_x5, neg_x5], dim=0)
                    pred_contrast = self.pred_layer(cat_x5)
                    pred_contrast = F.interpolate(pred_contrast, size=(H, W), mode='bilinear', align_corners=True)
            p5 = self.top_layer(weighted_x5)
        else:
            p5 = self.top_layer(x5)

        ########## Up-Sample ##########

        p4 = self._upsample_add(p5, self.latlayer4(x4)) 
        p4 = self.enlayer4(p4)
        p4_out = F.interpolate(self.dslayer4(p4), size=x3.shape[2:], mode='bilinear', align_corners=True)

        p3 = self._upsample_add(p4, self.latlayer3(x3)) 
        p3 = self.enlayer3(p3)
        p3_out = F.interpolate(self.dslayer3(p3), size=x2.shape[2:], mode='bilinear', align_corners=True)

        p2 = self._upsample_add(p3, self.latlayer2(x2)) 
        p2 = self.enlayer2(p2)
        p2_out = F.interpolate(self.dslayer2(p2), size=x1.shape[2:], mode='bilinear', align_corners=True)

        p1 = self._upsample_add(p2, self.latlayer1(x1)) 
        p1 = self.enlayer1(p1)
        p1_out = F.interpolate(self.dslayer1(p1), size=x.shape[2:], mode='bilinear', align_corners=True)

        _preds_may_be_masked = [p4_out, p3_out, p2_out, p1_out]
        scaled_preds = []
        for idx_out in range(len(_preds_may_be_masked)):
            if idx_out < len(_preds_may_be_masked) - config.loss_cls_mask_last_layers:
                scaled_preds.append(
                    _preds_may_be_masked[idx_out]
                )
            else:
                scaled_preds.append(
                    self.convs_out[idx_out - (len(_preds_may_be_masked) - config.loss_cls_mask_last_layers)](
                        _preds_may_be_masked[idx_out]
                    )
                )

        if 'cls_mask' in config.loss:
            pred_cls_masks = []
            input_features = [x, x1, x2, x3][:config.loss_cls_mask_last_layers]
            bb_lst = [self.bb.conv1, self.bb.conv2, self.bb.conv3, self.bb.conv4, self.bb.conv5]
            for idx_out in range(config.loss_cls_mask_last_layers):
                pred_cls_masks.append(
                    self.classifier(
                        self.avgpool(
                            nn.Sequential(*bb_lst[idx_out:])(
                                input_features[idx_out] * scaled_preds[-(idx_out+1)]
                            )
                        ).view(N, -1)
                    )
                )

        if self.training:
            if {'sal', 'cls', 'contrast', 'cls_mask'} == set(config.loss):
                return scaled_preds, pred_cls, pred_contrast, pred_cls_masks
            elif {'sal', 'cls', 'contrast'} == set(config.loss):
                return scaled_preds, pred_cls, pred_contrast
            elif {'sal', 'cls', 'cls_mask'} == set(config.loss):
                return scaled_preds, pred_cls, pred_cls_masks
            elif {'sal', 'cls'} == set(config.loss):
                return scaled_preds, pred_cls
            elif {'sal', 'contrast'} == set(config.loss):
                return scaled_preds, pred_contrast
            elif {'sal', 'cls_mask'} == set(config.loss):
                return scaled_preds, pred_cls_masks
            else:
                return scaled_preds
        else:
            return scaled_preds
