from random import seed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset import get_loader
import torchvision.utils as vutils

import torch.nn.functional as F
import pytorch_toolbelt.losses as PTL

from config import Config
from loss import saliency_structure_consistency, DSLoss
from util import generate_smoothed_gt

from evaluation.dataloader import EvalDataset
from evaluation.evaluator import Eval_thread

from models.GCoNet_plus import GCoNet_plus


# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='GCoNet_plus',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='Jigsaw2_DUTS',
                    type=str,
                    help="Options: 'DUTS_class'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--ckpt_dir', default=None, help='Temporary folder')

parser.add_argument('--testsets',
                    default='CoCA+CoSOD3k+CoSal2015',
                    type=str,
                    help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")

parser.add_argument('--val_dir',
                    default='tmp4val',
                    type=str,
                    help="Dir for saving tmp results for validation.")

args = parser.parse_args()


config = Config()

# Prepare dataset
if args.trainset == 'DUTS_class':
    root_dir = '/root/datasets/sod'
    train_img_path = os.path.join(root_dir, 'images/DUTS_class')
    train_gt_path = os.path.join(root_dir, 'gts/DUTS_class')
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              1,
                              max_num=config.batch_size,
                              istrain=True,
                              shuffle=False,
                              num_workers=8,
                              pin=True)
    train_img_path_seg = os.path.join(root_dir, 'images/coco-seg')
    train_gt_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    train_loader_seg = get_loader(
        train_img_path_seg,
        train_gt_path_seg,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=8,
        pin=True
    )
else:
    print('Unkonwn train dataset')
    print(args.dataset)

test_loaders = {}
for testset in args.testsets.split('+'):
    test_loader = get_loader(
        os.path.join('../../../datasets/sod', 'images', testset), os.path.join('../../../datasets/sod', 'gts', testset),
        args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True
    )
    test_loaders[testset] = test_loader

if config.rand_seed:
    set_seed(config.rand_seed)

# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_file = os.path.join(args.ckpt_dir, "log_loss.txt")
logger_loss_idx = 1

# Init model
device = torch.device("cuda")

model = GCoNet_plus()
model = model.to(device)
if config.lambda_adv:
    from adv import Discriminator
    disc = Discriminator(channels=1, img_size=args.size).to(device)
    optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.9, 0.99])
    Tensor = torch.cuda.FloatTensor if (True if torch.cuda.is_available() else False) else torch.FloatTensor
    adv_criterion = nn.BCELoss()

backbone_params = list(map(id, model.bb.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.parameters())

all_params = [{'params': base_params}, {'params': model.bb.parameters(), 'lr': config.lr * 0.01}]

# Setting optimizer
optimizer = optim.Adam(params=all_params, lr=config.lr, betas=[0.9, 0.99])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step_size, gamma=0.1)

# Why freeze the backbone?...
if config.freeze:
    for key, value in model.named_parameters():
        if 'bb' in key and 'bb.conv5.conv5_3' not in key:
            value.requires_grad = False


# log model and optimizer params
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
logger.info(scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
dsloss = DSLoss()


def main():
    val_measures = []

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            model.load_state_dict(torch.load(args.resume))
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            # logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            #     args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(epoch)
        if config.validation:
            measures = validate(model, test_loaders, args.testsets)
            val_measures.append(measures)
            print('Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()), np.max(np.array(val_measures)[:, 0]))
            )
        # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            path=args.ckpt_dir)
        if epoch >= args.epochs - config.val_last:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'ep{}.pth'.format(epoch)))
        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.ckpt_dir, weight_file) for weight_file in os.listdir(args.ckpt_dir) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))

def train(epoch):
    loss_log = AverageMeter()
    loss_log_triplet = AverageMeter()
    global logger_loss_idx
    model.train()
    FL = PTL.BinaryFocalLoss()

    for batch_idx, (batch, batch_seg) in enumerate(zip(train_loader, train_loader_seg)):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        cls_gts = torch.LongTensor(batch[-1]).to(device)
        
        gts_neg = torch.full_like(gts, 0.0)
        gts_cat = torch.cat([gts, gts_neg], dim=0)
        return_values = model(inputs)
        if {'sal', 'cls', 'contrast', 'cls_mask'} == set(config.loss):
            scaled_preds, pred_cls, pred_contrast, pred_cls_masks = return_values[:4]
        elif {'sal', 'cls', 'contrast'} == set(config.loss):
            scaled_preds, pred_cls, pred_contrast = return_values[:3]
        elif {'sal', 'cls', 'cls_mask'} == set(config.loss):
            scaled_preds, pred_cls, pred_cls_masks = return_values[:3]
        elif {'sal', 'cls'} == set(config.loss):
            scaled_preds, pred_cls = return_values[:2]
        elif {'sal', 'contrast'} == set(config.loss):
            scaled_preds, pred_contrast = return_values[:2]
        elif {'sal', 'cls_mask'} == set(config.loss):
            scaled_preds, pred_cls_masks = return_values[:2]
        else:
            scaled_preds = return_values[:1]
        norm_features = None
        if config.lambdas_sal_last['triplet']:
            norm_features = return_values[-1]
        scaled_preds = scaled_preds[-min(config.loss_sal_layers+int(bool(config.refine)), 4+int(bool(config.refine))):]

        # Tricks
        if config.lambdas_sal_last['triplet']:
            loss_sal, loss_triplet = dsloss(scaled_preds, gts, norm_features=norm_features, labels=cls_gts)
        else:
            loss_sal = dsloss(scaled_preds, gts)
        if config.label_smoothing:
            loss_sal = 0.5 * (loss_sal + dsloss(scaled_preds, generate_smoothed_gt(gts)))
        if config.self_supervision:
            H, W = inputs.shape[-2:]
            images_scale = F.interpolate(inputs, size=(H//4, W//4), mode='bilinear', align_corners=True)
            sal_scale = model(images_scale)[0][-1]
            atts = scaled_preds[-1]
            sal_s = F.interpolate(atts, size=(H//4, W//4), mode='bilinear', align_corners=True)
            loss_ss = saliency_structure_consistency(sal_scale.sigmoid(), sal_s.sigmoid())
            loss_sal += loss_ss * 0.3

        # Loss
        loss = 0
        # since there may be several losses for sal, the lambdas for them (lambdas_sal) are inside the loss.py
        loss_sal = loss_sal * 1
        loss += loss_sal
        if 'cls' in config.loss:
            loss_cls = F.cross_entropy(pred_cls, cls_gts) * config.lambda_cls
            loss += loss_cls
        if 'contrast' in config.loss:
            loss_contrast = FL(pred_contrast, gts_cat) * config.lambda_contrast
            loss += loss_contrast
        if 'cls_mask' in config.loss:
            loss_cls_mask = 0
            for pred_cls_mask in pred_cls_masks:
                loss_cls_mask += F.cross_entropy(pred_cls_mask, cls_gts) * config.lambda_cls_mask
            loss += loss_cls_mask
        if config.lambda_adv:
            # gen
            valid = Variable(Tensor(scaled_preds[-1].shape[0], 1).fill_(1.0), requires_grad=False)
            adv_loss_g = adv_criterion(disc(scaled_preds[-1]), valid)
            loss += adv_loss_g * config.lambda_adv

        loss_log.update(loss, inputs.size(0))
        if config.lambdas_sal_last['triplet']:
            loss_log_triplet.update(loss_triplet, inputs.size(0))

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
        inputs = batch_seg[0].to(device).squeeze(0)
        gts = batch_seg[1].to(device).squeeze(0)
        cls_gts = torch.LongTensor(batch_seg[-1]).to(device)

        gts_neg = torch.full_like(gts, 0.0)
        gts_cat = torch.cat([gts, gts_neg], dim=0)
        return_values = model(inputs)
        if {'sal', 'cls', 'contrast', 'cls_mask'} == set(config.loss):
            scaled_preds, pred_cls, pred_contrast, pred_cls_masks = return_values[:4]
        elif {'sal', 'cls', 'contrast'} == set(config.loss):
            scaled_preds, pred_cls, pred_contrast = return_values[:3]
        elif {'sal', 'cls', 'cls_mask'} == set(config.loss):
            scaled_preds, pred_cls, pred_cls_masks = return_values[:3]
        elif {'sal', 'cls'} == set(config.loss):
            scaled_preds, pred_cls = return_values[:2]
        elif {'sal', 'contrast'} == set(config.loss):
            scaled_preds, pred_contrast = return_values[:2]
        elif {'sal', 'cls_mask'} == set(config.loss):
            scaled_preds, pred_cls_masks = return_values[:2]
        else:
            scaled_preds = return_values[:1]
        norm_features = None
        if config.lambdas_sal_last['triplet']:
            norm_features = return_values[-1]
        scaled_preds = scaled_preds[-min(config.loss_sal_layers+int(bool(config.refine)), 4+int(bool(config.refine))):]

        # Tricks
        if config.lambdas_sal_last['triplet']:
            loss_sal, loss_triplet = dsloss(scaled_preds, gts, norm_features=norm_features, labels=cls_gts)
        else:
            loss_sal = dsloss(scaled_preds, gts)
        if config.label_smoothing:
            loss_sal = 0.5 * (loss_sal + dsloss(scaled_preds, generate_smoothed_gt(gts)))
        if config.self_supervision:
            H, W = inputs.shape[-2:]
            images_scale = F.interpolate(inputs, size=(H//4, W//4), mode='bilinear', align_corners=True)
            sal_scale = model(images_scale)[0][-1]
            atts = scaled_preds[-1]
            sal_s = F.interpolate(atts, size=(H//4, W//4), mode='bilinear', align_corners=True)
            loss_ss = saliency_structure_consistency(sal_scale.sigmoid(), sal_s.sigmoid())
            loss_sal += loss_ss * 0.3

        # Loss
        # loss = 0
        # since there may be several losses for sal, the lambdas for them (lambdas_sal) are inside the loss.py
        loss_sal = loss_sal * 1
        loss += loss_sal
        if 'cls' in config.loss:
            loss_cls = F.cross_entropy(pred_cls, cls_gts) * config.lambda_cls
            loss += loss_cls
        if 'contrast' in config.loss:
            loss_contrast = FL(pred_contrast, gts_cat) * config.lambda_contrast
            loss += loss_contrast
        if 'cls_mask' in config.loss:
            loss_cls_mask = 0
            for pred_cls_mask in pred_cls_masks:
                loss_cls_mask += F.cross_entropy(pred_cls_mask, cls_gts) * config.lambda_cls_mask
            loss += loss_cls_mask
        if config.lambda_adv:
            # gen
            valid = Variable(Tensor(scaled_preds[-1].shape[0], 1).fill_(1.0), requires_grad=False)
            adv_loss_g = adv_criterion(disc(scaled_preds[-1]), valid)
            loss += adv_loss_g * config.lambda_adv

        loss_log.update(loss, inputs.size(0))
        if config.lambdas_sal_last['triplet']:
            loss_log_triplet.update(loss_triplet, inputs.size(0))
        with open(logger_loss_file, 'a') as f:
            f.write('step {}, {}\n'.format(logger_loss_idx, loss))
        logger_loss_idx += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

        if config.lambda_adv and batch_idx % 5 == 0:
            # disc
            fake = Variable(Tensor(scaled_preds[-1].shape[0], 1).fill_(0.0), requires_grad=False)
            optimizer_d.zero_grad()
            adv_loss_real = adv_criterion(disc(gts), valid)
            adv_loss_fake = adv_criterion(disc(scaled_preds[-1].detach()), fake)
            adv_loss_d = (adv_loss_real + adv_loss_fake) / 2 * 1.
            adv_loss_d.backward()
            optimizer_d.step()

        # Logger
        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}]'.format(epoch, args.epochs, batch_idx, len(train_loader))
            info_loss = 'Train Loss: loss_sal: {:.3f}'.format(loss_sal)
            if 'cls' in config.loss:
                info_loss += ', loss_cls: {:.3f}'.format(loss_cls)
            if 'cls_mask' in config.loss:
                info_loss += ', loss_cls_mask: {:.3f}'.format(loss_cls_mask)
            if 'contrast' in config.loss:
                info_loss += ', loss_contrast: {:.3f}'.format(loss_contrast)
            if config.lambda_adv:
                info_loss += ', loss_adv: {:.3f}, loss_adv_disc: {:.3f}'.format(adv_loss_g, adv_loss_d)
            if config.lambdas_sal_last['triplet']:
                info_loss += ', loss_triplet: {:.3f}'.format(loss_triplet)
            info_loss += ', Loss_total: {loss.val:.3f} ({loss.avg:.3f})  '.format(loss=loss_log)
            logger.info(''.join((info_progress, info_loss)))
    scheduler.step()
    info_loss = '@==Final== Epoch[{0}/{1}]  Train Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=loss_log)
    if config.lambdas_sal_last['triplet']:
        info_loss += 'Triplet Loss: {loss.avg:.3f}  '.format(loss=loss_log_triplet)
    logger.info(info_loss)

    return loss_log.avg


def validate(model, test_loaders, testsets):
    model.eval()

    testsets = testsets.split('+')
    measures = []
    for testset in testsets[:1]:
        print('Validating {}...'.format(testset))
        test_loader = test_loaders[testset]
        
        saved_root = os.path.join(args.val_dir, testset)

        for batch in test_loader:
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs)[-1]

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                if config.db_output_refiner or (not config.refine and config.db_output_decoder):
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
                else:
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True).sigmoid()
                save_tensor_img(res, os.path.join(saved_root, subpath))

        eval_loader = EvalDataset(
            saved_root,                                                             # preds
            os.path.join('/root/datasets/sod/gts', testset)                     # GT
        )
        evaler = Eval_thread(eval_loader, cuda=True)
        # Use S_measure for validation
        s_measure = evaler.Eval_Smeasure()
        if s_measure > config.val_measures['Smeasure']['CoCA'] and 0:
            # TODO: evluate others measures if s_measure is very high.
            e_max = evaler.Eval_Emeasure().max().item()
            f_max = evaler.Eval_fmeasure().max().item()
            print('Emax: {:4.f}, Fmax: {:4.f}'.format(e_max, f_max))
        measures.append(s_measure)

    model.train()
    return measures

if __name__ == '__main__':
    main()
