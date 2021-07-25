import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from config import Config


def pairwise_distance_torch(embeddings, device=torch.device("cuda")):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def TripletSemiHardLoss(y_pred, y_true, device=torch.device("cuda"), margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    labels = torch.reshape(labels, [labels.shape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, ipt, target, **kwargs):
        return TripletSemiHardLoss(ipt, target, margin=self.margin)


class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        # return IoU/b
        return IoU


class ThrReg_loss(torch.nn.Module):
    def __init__(self):
        super(ThrReg_loss, self).__init__()

    def forward(self, pred, gt=None):
        return torch.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))


class DSLoss(nn.Module):
    """
    IoU loss for outputs in [1:] scales.
    """
    def __init__(self):
        super(DSLoss, self).__init__()
        self.config = Config()
        self.lambdas_sal_last = self.config.lambdas_sal_last
        self.lambdas_sal_others = self.config.lambdas_sal_others
        self.triplet_loss = ['vanilla', 'semi_hard'][0]

        self.criterions_last = {}
        if 'bce' in self.lambdas_sal_last and self.lambdas_sal_last['bce']:
            self.criterions_last['bce'] = nn.BCELoss()
        if 'iou' in self.lambdas_sal_last and self.lambdas_sal_last['iou']:
            self.criterions_last['iou'] = IoU_loss()
        if 'ssim' in self.lambdas_sal_last and self.lambdas_sal_last['ssim']:
            self.criterions_last['ssim'] = SSIMLoss()
        if 'mse' in self.lambdas_sal_last and self.lambdas_sal_last['mse']:
            self.criterions_last['mse'] = nn.MSELoss()
        if 'reg' in self.lambdas_sal_last and self.lambdas_sal_last['reg']:
            self.criterions_last['reg'] = ThrReg_loss()
        if 'triplet' in self.lambdas_sal_last and self.lambdas_sal_last['triplet']:
            margin = self.config.triplet_loss_margin
            if self.triplet_loss == 'vanilla':
                self.criterion_triplet = nn.TripletMarginLoss(margin=margin)
            elif self.criterion_triplet == 'semi_hard':
                self.criterion_triplet = TripletLoss(margin=margin)

        self.criterions_others = {}
        if 'bce' in self.lambdas_sal_others and self.lambdas_sal_others['bce']:
            self.criterions_others['bce'] = nn.BCELoss()
        if 'iou' in self.lambdas_sal_others and self.lambdas_sal_others['iou']:
            self.criterions_others['iou'] = IoU_loss()
        if 'ssim' in self.lambdas_sal_others and self.lambdas_sal_others['ssim']:
            self.criterions_others['ssim'] = SSIMLoss()
        if 'mse' in self.lambdas_sal_others and self.lambdas_sal_others['mse']:
            self.criterions_others['mse'] = nn.MSELoss()

    def forward(self, scaled_preds, gt, norm_features=None, labels=None):
        loss = 0
        for idx_output, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != gt.shape:
                pred_lvl = nn.functional.interpolate(pred_lvl, size=gt.shape[2:], mode='bilinear', align_corners=True)
            if idx_output == len(scaled_preds) - 1:
                if not(self.config.db_output_refiner or (not self.config.refine and self.config.db_output_decoder)):
                    pred_lvl = pred_lvl.sigmoid()
                for criterion_name, criterion in self.criterions_last.items():
                    loss += criterion(pred_lvl, gt) * self.lambdas_sal_last[criterion_name]
                # loss_outside = self.criterions_last['iou'](pred_lvl * (1 - gt), gt * (1 - gt)) * self.lambdas_sal_last['iou'] * 2
                # loss_inside = self.criterions_last['bce'](pred_lvl * gt, gt) * self.lambdas_sal_last['bce'] * 2
                # loss_inside += self.criterions_last['mse'](pred_lvl * gt, gt) * self.lambdas_sal_last['mse'] * 2
                # loss += (loss_outside + loss_inside)
            else:
                if not (self.config.refine and self.config.db_output_decoder and idx_output == len(scaled_preds) - 2):
                    pred_lvl = pred_lvl.sigmoid()
                for criterion_name, criterion in self.criterions_others.items():
                    loss += criterion(pred_lvl, gt) * self.lambdas_sal_others[criterion_name]
        if self.lambdas_sal_last['triplet'] and norm_features is not None:
            triplet_loss = 0
            for norm_feature in norm_features:
                # vanilla triplet loss in PyTorch
                if self.triplet_loss == 'vanilla':
                    num_feature_per_group = norm_feature.shape[0] // 2
                    feat_A = norm_feature[:num_feature_per_group]
                    feat_B = norm_feature[num_feature_per_group:]
                    # A/2 - A/2 - B/2
                    loss_triplet_ancA = self.criterion_triplet(feat_A[:num_feature_per_group//2], feat_A[-(num_feature_per_group//2):], feat_B[:num_feature_per_group//2])
                    loss_triplet_ancB = self.criterion_triplet(feat_B[:num_feature_per_group//2], feat_B[-(num_feature_per_group//2):], feat_A[:num_feature_per_group//2])
                    triplet_loss += (loss_triplet_ancA + loss_triplet_ancB)
                elif self.triplet_loss == 'semi_hard':
                    triplet_loss += self.criterion_triplet(norm_feature, labels)
            triplet_loss = triplet_loss * self.config.lambdas_sal_last['triplet'] / len(norm_features)
            loss += triplet_loss
            return loss, triplet_loss
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def saliency_structure_consistency(x, y):
    ssim = torch.mean(SSIM(x,y))
    return ssim
