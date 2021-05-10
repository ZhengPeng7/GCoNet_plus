from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from util import save_tensor_img, Logger
from tqdm import tqdm
from torch import nn
import os
from criterion import Eval
import argparse
import numpy as np


def main(args):
    # Init model

    device = torch.device("cuda")
    exec('from models import ' + args.model)
    model = eval(args.model+'()')
    model = model.to(device)
    ginet_dict = torch.load(args.ckpt)

    model.to(device)
    model.ginet.load_state_dict(ginet_dict)

    model.eval()
    model.set_mode('test')

    tensor2pil = transforms.ToPILImage()

    for testset in args.testsets.split('+'):
        print('Testing {}...'.format(testset))
        root_dir = '../../../datasets/sod'
        if testset == 'CoCA':
            test_img_path = os.path.join(root_dir, 'images/CoCA')
            test_gt_path = os.path.join(root_dir, 'gts/CoCA')
            saved_root = os.path.join(args.pred_dir, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = os.path.join(root_dir, 'images/CoSOD3k')
            test_gt_path = os.path.join(root_dir, 'gts/CoSOD3k')
            saved_root = os.path.join(args.pred_dir, 'CoSOD3k')
        elif testset == 'CoSal2015':
            test_img_path = os.path.join(root_dir, 'images/CoSal2015')
            test_gt_path = os.path.join(root_dir, 'gts/CoSal2015')
            saved_root = os.path.join(args.pred_dir, 'CoSal2015')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)
        
        test_loader = get_loader(
            test_img_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8, pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            
            scaled_preds = model(inputs)[-1]

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear', align_corners=True)
                save_tensor_img(res, os.path.join(saved_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet',
                        type=str,
                        help="Options: '', ''")
    parser.add_argument('--testsets',
                       default='CoCA+CoSOD3k+CoSal2015',
                       type=str,
                       help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--ckpt', default='./ckpt/gconet_final.pth', type=str, help='model folder')
    parser.add_argument('--pred_dir', default='/home/pz1/datasets/sod/preds/GCoNet_ext', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)
