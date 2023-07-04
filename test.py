import os
import argparse
from tqdm import tqdm
import torch
from torch import nn

from dataset import get_loader
from models.GCoNet_plus import GCoNet_plus
from util import save_tensor_img
from config import Config


def main(args):
    # Init model
    config = Config()

    device = torch.device("cuda")
    model = GCoNet_plus()
    model = model.to(device)
    print('Testing with model {}'.format(args.ckpt))
    gconet_dict = torch.load(args.ckpt)

    model.to(device)
    model.load_state_dict(gconet_dict)

    model.eval()

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


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet_plus',
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
    parser.add_argument('--ckpt', default='./ckpt/GCoNet_plus/final.pth', type=str, help='model folder')
    parser.add_argument('--pred_dir', default='/root/datasets/sod/preds/GCoNet_plus', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)
