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
        lateral_channels_in = [512, 512, 256, 128, 64] if 'vgg16' in bb else [2048, 1024, 512, 256, 64]


        ch_decoder = lateral_channels_in[0]//2//channel_scale
        self.top_layer = ResBlk(lateral_channels_in[0], ch_decoder)
        self.enlayer5 = ResBlk(ch_decoder, ch_decoder)
        if config.conv_after_itp:
            self.dslayer5 = DSLayer(ch_decoder, ch_decoder)
        self.latlayer5 = ResBlk(lateral_channels_in[1], ch_decoder) if config.complex_lateral_connection else nn.Conv2d(lateral_channels_in[1], ch_decoder, 1, 1, 0)

        ch_decoder //= 2
        self.enlayer4 = ResBlk(ch_decoder*2, ch_decoder)
        if config.conv_after_itp:
            self.dslayer4 = DSLayer(ch_decoder, ch_decoder)
        self.latlayer4 = ResBlk(lateral_channels_in[2], ch_decoder) if config.complex_lateral_connection else nn.Conv2d(lateral_channels_in[2], ch_decoder, 1, 1, 0)
        if config.output_number >= 4:
            self.conv_out4 = nn.Sequential(nn.Conv2d(ch_decoder, 32, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1, 1, 0))

        ch_decoder //= 2
        self.enlayer3 = ResBlk(ch_decoder*2, ch_decoder)
        if config.conv_after_itp:
            self.dslayer3 = DSLayer(ch_decoder, ch_decoder)
        self.latlayer3 = ResBlk(lateral_channels_in[3], ch_decoder) if config.complex_lateral_connection else nn.Conv2d(lateral_channels_in[3], ch_decoder, 1, 1, 0)
        if config.output_number >= 3:
            self.conv_out3 = nn.Sequential(nn.Conv2d(ch_decoder, 32, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1, 1, 0))

        ch_decoder //= 2
        self.enlayer2 = ResBlk(ch_decoder*2, ch_decoder)
        if config.conv_after_itp:
            self.dslayer2 = DSLayer(ch_decoder, ch_decoder)
        self.latlayer2 = ResBlk(lateral_channels_in[4], ch_decoder) if config.complex_lateral_connection else nn.Conv2d(lateral_channels_in[4], ch_decoder, 1, 1, 0)
        if config.output_number >= 2:
            self.conv_out2 = nn.Sequential(nn.Conv2d(ch_decoder, 32, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1, 1, 0))

        self.enlayer1 = ResBlk(ch_decoder, ch_decoder)
        self.conv_out1 = nn.Sequential(nn.Conv2d(ch_decoder, 1, 1, 1, 0))

        if config.GAM:
            self.co_x5 = CoAttLayer(channel_in=lateral_channels_in[0])

        if 'contrast' in config.loss:
            self.pred_layer = half_DSLayer(lateral_channels_in[0])

        if {'cls', 'cls_mask'} & set(config.loss):
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(lateral_channels_in[0], 291)       # DUTS_class has 291 classes
            for layer in [self.classifier]:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        ########## Encoder ##########

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

        ########## Decoder ##########
        scaled_preds = []
        p5 = self.enlayer5(p5)
        p5 = F.interpolate(p5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        if config.conv_after_itp:
            p5 = self.dslayer5(p5)
        p4 = p5 + self.latlayer5(x4)

        p4 = self.enlayer4(p4)
        p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        if config.conv_after_itp:
            p4 = self.dslayer4(p4)
        if config.output_number >= 4:
            p4_out = self.conv_out4(p4)
            scaled_preds.append(p4_out)
        p3 = p4 + self.latlayer4(x3)

        p3 = self.enlayer3(p3)
        p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        if config.conv_after_itp:
            p3 = self.dslayer3(p3)
        if config.output_number >= 3:
            p3_out = self.conv_out3(p3)
            scaled_preds.append(p3_out)
        p2 = p3 + self.latlayer3(x2)

        p2 = self.enlayer2(p2)
        p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        if config.conv_after_itp:
            p2 = self.dslayer2(p2)
        if config.output_number >= 2:
            p2_out = self.conv_out2(p2)
            scaled_preds.append(p2_out)
        p1 = p2 + self.latlayer2(x1)

        p1 = self.enlayer1(p1)
        p1 = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=True)
        p1_out = self.conv_out1(p1)
        scaled_preds.append(p1_out)

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
