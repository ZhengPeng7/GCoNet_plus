import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from config import Config


class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super(ResBlk, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(64, channel_out, 3, 1, 1)

    def forward(self, x):
        x = self.relu_in(self.conv_in(x))
        x = self.conv_out(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, channel_in=64, sigmoid=True):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(channel_in, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        ) if sigmoid else nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
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
    def __init__(self, input_channels=512):

        super(GAM, self).__init__()
        self.query_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.key_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 

        self.scale = 1.0 / (input_channels ** 0.5)

        self.conv6 = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 

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
    def __init__(self, channel_in=512, num_heads=8):
        super(MHA, self).__init__()
        self.norm_layer = nn.LayerNorm(channel_in)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=channel_in, num_heads=num_heads)

    def forward(self, x):
        bs, ch, h, w = x.shape
        y = x.view(bs, -1, ch)
        y = self.multi_head_attention(y, y, y)[0]
        y = y.view(bs, h, w, ch).transpose(-1, -2).transpose(-2, -3)
        x = x + y
        return x


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

class CoAttLayer(nn.Module):
    def __init__(self, input_channels=512):
        super(CoAttLayer, self).__init__()

        self.all_attention = eval(Config().relation_module + '(input_channels)')
        self.conv_output = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.conv_transform = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0) 
        self.fc_transform = nn.Linear(input_channels, input_channels)

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
