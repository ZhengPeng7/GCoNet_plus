import os
import time
import json

import numpy as np
from scipy.io import savemat
import torch
from torchvision import transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Eval_thread():
    def __init__(self, loader, method='', dataset='', output_dir='', epoch='', cuda=True):
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.output_dir = output_dir
        self.epoch = epoch.split('ep')[-1]
        self.logfile = os.path.join(output_dir, 'result.txt')
        self.dataset2smeasure_bottom_bound = {'CoCA': 0.673, 'CoSOD3k': 0.802, 'CoSal2015': 0.845}      # S_measures of GCoNet

    def run(self, AP=False, AUC=False, save_metrics=False, continue_eval=True):
        Res = {}
        start_time = time.time()

        if continue_eval:
            s = self.Eval_Smeasure()
            if s > self.dataset2smeasure_bottom_bound[self.dataset]:
                mae = self.Eval_mae()
                Em = self.Eval_Emeasure()
                max_e = Em.max().item()
                mean_e = Em.mean().item()
                Em = Em.cpu().numpy()
                Fm, prec, recall = self.Eval_fmeasure()
                max_f = Fm.max().item()
                mean_f = Fm.mean().item()
                Fm = Fm.cpu().numpy()
            else:
                mae = 1
                Em = torch.zeros(255).cpu().numpy()
                max_e = 0
                mean_e = 0
                Fm, prec, recall = 0, 0, 0
                max_f = 0
                mean_f = 0
                continue_eval = False
        else:
            s = 0
            mae = 1
            Em = torch.zeros(255).cpu().numpy()
            max_e = 0
            mean_e = 0
            Fm, prec, recall = 0, 0, 0
            max_f = 0
            mean_f = 0
            continue_eval = False


        if AP:
            prec = prec.cpu().numpy()
            recall = recall.cpu().numpy()
            avg_p = self.Eval_AP(prec, recall)

        if AUC:
            auc, TPR, FPR = self.Eval_auc()
            TPR = TPR.cpu().numpy()
            FPR = FPR.cpu().numpy()

        if save_metrics:
            os.makedirs(os.path.join(self.output_dir, self.method, self.epoch), exist_ok=True)
            Res['Sm'] = s
            if s > self.dataset2smeasure_bottom_bound[self.dataset]:
                Res['MAE'] = mae
                Res['MaxEm'] = max_e
                Res['MeanEm'] = mean_e
                Res['Em'] = Em
                Res['Fm'] = Fm
            else:
                Res['MAE'] = 1
                Res['MaxEm'] = 0
                Res['MeanEm'] = 0
                Res['Em'] = torch.zeros(255).cpu().numpy()
                Res['Fm'] = 0

            if AP:
                Res['MaxFm'] = max_f
                Res['MeanFm'] = mean_f
                Res['AP'] = avg_p
                Res['Prec'] = prec
                Res['Recall'] = recall

            if AUC:
                Res['AUC'] = auc
                Res['TPR'] = TPR
                Res['FPR'] = FPR

            os.makedirs(os.path.join(self.output_dir, self.method, self.epoch), exist_ok=True)
            savemat(os.path.join(self.output_dir, self.method, self.epoch, self.dataset + '.mat'), Res)

        info = '{} ({}): {:.4f} max-Emeasure || {:.4f} S-measure  || {:.4f} max-fm || {:.4f} mae || {:.4f} mean-Emeasure || {:.4f} mean-fm'.format(
            self.dataset, self.method+'-ep{}'.format(self.epoch), max_e, s, max_f, mae, mean_e, mean_f
        )
        if AP:
            info += ' || {:.4f} AP'.format(avg_p)
        if AUC:
            info += ' || {:.4f} AUC'.format(auc)
        info += '.'
        self.LOG(info + '\n')

        return '[cost:{:.4f}s] '.format(time.time() - start_time) + info, continue_eval

    def Eval_mae(self):
        if self.epoch:
            print('Evaluating MAE...')
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item()

    def Eval_fmeasure(self):
        print('Evaluating FMeasure...')
        beta2 = 0.3
        avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                prec, recall = self._eval_pr(pred, gt, 255)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0  # for Nan
                avg_f += f_score
                avg_p += prec
                avg_r += recall
                img_num += 1.0
            Fm = avg_f / img_num
            avg_p = avg_p / img_num
            avg_r = avg_r / img_num
            return Fm, avg_p, avg_r

    def Eval_auc(self):
        print('Evaluating AUC...')

        avg_tpr, avg_fpr, avg_auc, img_num = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                TPR, FPR = self._eval_roc(pred, gt, 255)
                avg_tpr += TPR
                avg_fpr += FPR
                img_num += 1.0
            avg_tpr = avg_tpr / img_num
            avg_fpr = avg_fpr / img_num

            sorted_idxes = torch.argsort(avg_fpr)
            avg_tpr = avg_tpr[sorted_idxes]
            avg_fpr = avg_fpr[sorted_idxes]
            avg_auc = torch.trapz(avg_tpr, avg_fpr)

            return avg_auc.item(), avg_tpr, avg_fpr

    def Eval_Emeasure(self):
        print('Evaluating EMeasure...')
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            Em = torch.zeros(255)
            if self.cuda:
                Em = Em.cuda()
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                Em += self._eval_e(pred, gt, 255)
                img_num += 1.0

            Em /= img_num
            return Em

    def select_by_Smeasure(self, bar=0.9, loader_comp=None, bar_comp=0.1):
        print('Evaluating SMeasure...')
        good_ones = []
        good_ones_comp = []
        good_ones_gt = []
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for (pred, gt, predpath, gtpath), (pred_comp, gt_comp, predpath_comp) in zip(self.loader, loader_comp):
                # pred X gt
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    gt[gt >= 0.5] = 1
                    gt[gt < 0.5] = 0
                    Q = alpha * self._S_object(
                        pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                img_num += 1.0
                avg_q += Q.item()
                # pred_comp X gt
                if self.cuda:
                    pred_comp = trans(pred_comp).cuda()
                    pred_comp = (pred_comp - torch.min(pred_comp)) / (torch.max(pred_comp) -
                                                       torch.min(pred_comp) + 1e-20)
                    gt_comp = trans(gt_comp).cuda()
                else:
                    pred_comp = trans(pred_comp)
                    pred_comp = (pred_comp - torch.min(pred_comp)) / (torch.max(pred_comp) -
                                                       torch.min(pred_comp) + 1e-20)
                    gt_comp = trans(gt_comp)
                y = gt_comp.mean()
                if y == 0:
                    x = pred_comp.mean()
                    Q_comp = 1.0 - x
                elif y == 1:
                    x = pred_comp.mean()
                    Q_comp = x
                else:
                    gt_comp[gt_comp >= 0.5] = 1
                    gt_comp[gt_comp < 0.5] = 0
                    Q_comp = alpha * self._S_object(
                        pred_comp, gt_comp) + (1 - alpha) * self._S_region(pred_comp, gt_comp)
                    if Q_comp.item() < 0:
                        Q_comp = torch.FloatTensor([0.0])
                if Q.item() > bar and (Q.item() - Q_comp.item()) > bar_comp:
                    good_ones.append(predpath)
                    good_ones_comp.append(predpath_comp)
                    good_ones_gt.append(gtpath)
            avg_q /= img_num
            return avg_q, good_ones, good_ones_comp, good_ones_gt

    def Eval_Smeasure(self):
        print('Evaluating SMeasure...')
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                    gt = trans(gt)
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    gt[gt >= 0.5] = 1
                    gt[gt < 0.5] = 0
                    Q = alpha * self._S_object(
                        pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                img_num += 1.0
                avg_q += Q.item()
            avg_q /= img_num
            return avg_q

    def LOG(self, output):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.logfile, 'a') as f:
            f.write(output)

    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _eval_roc(self, y_pred, y, num):
        if self.cuda:
            TPR, FPR = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            TPR, FPR = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            fp = (y_temp * (1 - y)).sum()
            tn = ((1 - y_temp) * (1 - y)).sum()
            fn = ((1 - y_temp) * y).sum()

            TPR[i] = tp / (tp + fn + 1e-20)
            FPR[i] = fp / (fp + tn + 1e-20)

        return TPR, FPR

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0, cols)).cuda().float()
                j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    def Eval_AP(self, prec, recall):
        # Ref:
        # https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54
        print('Evaluating AP...')
        ap_r = np.concatenate(([0.], recall, [1.]))
        ap_p = np.concatenate(([0.], prec, [0.]))
        sorted_idxes = np.argsort(ap_r)
        ap_r = ap_r[sorted_idxes]
        ap_p = ap_p[sorted_idxes]
        count = ap_r.shape[0]

        for i in range(count - 1, 0, -1):
            ap_p[i - 1] = max(ap_p[i], ap_p[i - 1])

        i = np.where(ap_r[1:] != ap_r[:-1])[0]
        ap = np.sum((ap_r[i + 1] - ap_r[i]) * ap_p[i + 1])
        return ap
