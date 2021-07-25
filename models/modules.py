import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from config import Config


config = Config()


class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super(ResBlk, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(64, channel_out, 3, 1, 1)
        if config.use_bn:
            self.bn_in = nn.BatchNorm2d(64)
            self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        if config.use_bn:
            x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.conv_out(x)
        if config.use_bn:
            x = self.bn_out(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, channel_in=64, channel_out=1, activation_out='relu'):
        super(DSLayer, self).__init__()
        self.activation_out = activation_out
        self.conv1 = nn.Conv2d(channel_in, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        if activation_out:
            self.pred_conv = nn.Conv2d(64, channel_out, kernel_size=1, stride=1, padding=0)
            self.pred_relu = nn.ReLU(inplace=True)
        else:
            self.pred_conv = nn.Conv2d(64, channel_out, kernel_size=1, stride=1, padding=0)

        if config.use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.pred_bn = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv1(x)
        if config.use_bn:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if config.use_bn:
            x = self.bn2(x)
        x = self.relu2(x)

        x = self.pred_conv(x)
        if config.use_bn:
            x = self.pred_bn(x)
        if self.activation_out:
            x = self.pred_relu(x)
        return x


class half_DSLayer(nn.Module):
    def __init__(self, channel_in=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(channel_in, int(channel_in//4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(channel_in//4), 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class CoAttLayer(nn.Module):
    def __init__(self, channel_in=512):
        super(CoAttLayer, self).__init__()

        self.all_attention = eval(Config().relation_module + '(channel_in)')
        self.conv_output = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.conv_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
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


class ICE(nn.Module):
    # The Integrity Channel Enhancement (ICE) module
    # _X means in X-th column
    def __init__(self, channel_in=512):
        super(ICE, self).__init__()
        self.conv_1 = nn.Conv2d(channel_in, channel_in, 3, 1, 1)
        self.conv_2 = nn.Conv1d(channel_in, channel_in, 3, 1, 1)
        self.conv_3 = nn.Conv2d(channel_in*3, channel_in, 3, 1, 1)

        self.fc_2 = nn.Linear(channel_in, channel_in)
        self.fc_3 = nn.Linear(channel_in, channel_in)

    def forward(self, x):
        x_1, x_2, x_3 = x, x, x

        x_1 = x_1 * x_2 * x_3
        x_2 = x_1 + x_2 + x_3
        x_3 = torch.cat((x_1, x_2, x_3), dim=1)

        V = self.conv_1(x_1)

        bs, c, h, w = x_2.shape
        K = self.conv_2(x_2.view(bs, c, h*w))
        Q_prime = self.conv_3(x_3)
        Q_prime = torch.norm(Q_prime, dim=(-2, -1)).view(bs, c, 1, 1)
        Q_prime = Q_prime.view(bs, -1)
        Q_prime = self.fc_3(Q_prime)
        Q_prime = torch.softmax(Q_prime, dim=-1)
        Q_prime = Q_prime.unsqueeze(1)

        Q = torch.matmul(Q_prime, K)

        x_2 = torch.nn.functional.cosine_similarity(K, Q, dim=-1)
        x_2 = torch.sigmoid(x_2)
        x_2 = self.fc_2(x_2)
        x_2 = x_2.unsqueeze(-1).unsqueeze(-1)
        x_1 = V * x_2 + V

        return x_1


class GAM(nn.Module):
    def __init__(self, channel_in=512):

        super(GAM, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (channel_in ** 0.5)

        self.conv6 = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0) 

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


class MHA(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model=512, d_k=512, d_v=512, h=8, dropout=.1, channel_in=512):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MHA, self).__init__()
        self.query_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.key_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.value_transform = nn.Conv2d(channel_in, channel_in, kernel_size=1, stride=1, padding=0)
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        B, C, H, W = x.size()
        queries = self.query_transform(x).view(B, -1, C)
        keys = self.query_transform(x).view(B, -1, C)
        values = self.query_transform(x).view(B, -1, C)

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out).view(B, C, H, W)  # (b_s, nq, d_model)
        return out


class NonLocal(nn.Module):
    def __init__(self, channel_in=512, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(NonLocal, self).__init__()

        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.channel_in = channel_in
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = channel_in // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(self.channel_in, self.inter_channels, 1, 1, 0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(self.inter_channels, self.channel_in, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.channel_in)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(self.inter_channels, self.channel_in, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(self.channel_in, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.channel_in, self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class DBHead(nn.Module):
    def __init__(self, channel_in=32, channel_out=1, k=config.db_k):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, 3, 1, 1),
            *[nn.BatchNorm2d(channel_in), nn.ReLU(inplace=True)] if config.use_bn else nn.ReLU(inplace=True),
            nn.Conv2d(channel_in, channel_in, 3, 1, 1),
            *[nn.BatchNorm2d(channel_in), nn.ReLU(inplace=True)] if config.use_bn else nn.ReLU(inplace=True),
            nn.Conv2d(channel_in, channel_out, 1, 1, 0),
            nn.Sigmoid()
        )

        self.thresh = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, 3, padding=1),
            *[nn.BatchNorm2d(channel_in), nn.ReLU(inplace=True)] if config.use_bn else nn.ReLU(inplace=True),
            nn.Conv2d(channel_in, channel_in, 3, 1, 1),
            *[nn.BatchNorm2d(channel_in), nn.ReLU(inplace=True)] if config.use_bn else nn.ReLU(inplace=True),
            nn.Conv2d(channel_in, channel_out, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        return binary_maps

    def step_function(self, x, y):
        if config.db_k_alpha != 1:
            z = x - y
            mask_neg_inv = 1 - 2 * (z < 0)
            a = torch.exp(-self.k * (torch.pow(z * mask_neg_inv + 1e-16, 1/config.k_alpha) * mask_neg_inv))
        else:
            a = torch.exp(-self.k * (x - y))
        if torch.isinf(a).any():
            a = torch.exp(-50 * (x - y))
        return torch.reciprocal(1 + a)


class RefUnet(nn.Module):
    # Refinement
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()
        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)
        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        if config.use_bn:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        if config.use_bn:
            self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        if config.use_bn:
            self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        if config.use_bn:
            self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        #####
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        if config.use_bn:
            self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        #####
        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        if config.use_bn:
            self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        if config.use_bn:
            self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        if config.use_bn:
            self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        if config.use_bn:
            self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if config.db_output_refiner:
            self.db_output_refiner = DBHead(64)


    def forward(self, x):
        hx = x
        hx = self.conv1(self.conv0(hx))
        if config.use_bn:
            hx = self.bn1(hx)
        hx1 = self.relu1(hx)
        hx = self.conv2(self.pool1(hx1))
        if config.use_bn:
            hx = self.bn2(hx)
        hx2 = self.relu2(hx)
        hx = self.conv3(self.pool2(hx2))
        if config.use_bn:
            hx = self.bn3(hx)
        hx3 = self.relu3(hx)
        hx = self.conv4(self.pool3(hx3))
        if config.use_bn:
            hx = self.bn4(hx)
        hx4 = self.relu4(hx)
        hx = self.conv5(self.pool4(hx4))
        if config.use_bn:
            hx = self.bn5(hx)
        hx5 = self.relu5(hx)
        hx = self.upscore2(hx5)
        d4 = self.conv_d4(torch.cat((hx, hx4), 1))
        if config.use_bn:
            d4 = self.bn_d4(d4)
        d4 = self.relu_d4(d4)
        hx = self.upscore2(d4)
        d3 = self.conv_d3(torch.cat((hx, hx3), 1))
        if config.use_bn:
            d3 = self.bn_d3(d3)
        d3 = self.relu_d3(d3)
        hx = self.upscore2(d3)
        d2 = self.conv_d2(torch.cat((hx, hx2), 1))
        if config.use_bn:
            d2 = self.bn_d2(d2)
        d2 = self.relu_d2(d2)
        hx = self.upscore2(d2)
        d1 = self.conv_d1(torch.cat((hx, hx1), 1))
        if config.use_bn:
            d1 = self.bn_d1(d1)
        d1 = self.relu_d1(d1)
        if config.db_output_refiner:
            x = self.db_output_refiner(d1)
        else:
            residual = self.conv_d0(d1)
            x = x + residual
        return x
