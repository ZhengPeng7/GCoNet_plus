import os
import argparse
import numpy as np
import cv2

from evaluator import Eval_thread
from dataloader import EvalDataset

import sys
sys.path.append('..')


def main(cfg):
    dataset_names = cfg.datasets.split('+')
    root_dir_predictions = [dr for dr in os.listdir('.') if 'gconet_' in dr]
    root_dir_prediction_comp = cfg.gt_dir.replace('/gts', '/gconet')
    print('root_dir_predictions:', root_dir_predictions)
    root_dir_prediction = root_dir_predictions[0]
    root_dir_good_ones = 'good_ones'
    for dataset in dataset_names:
        dir_prediction = os.path.join(root_dir_prediction, dataset)
        dir_prediction_comp = os.path.join(root_dir_prediction_comp, dataset)
        dir_gt = os.path.join(cfg.gt_dir, dataset)
        loader = EvalDataset(
            dir_prediction,        # preds
            dir_gt,                   # GT
            return_predpath=True,
            return_gtpath=True
        )
        loader_comp = EvalDataset(
            dir_prediction_comp,        # preds
            dir_gt,                   # GT
            return_predpath=True
        )
        print('Selecting predictions from {}'.format(dir_prediction))
        thread = Eval_thread(loader, cuda=cfg.cuda)
        s_measure, good_ones, good_ones_comp, good_ones_gt = thread.select_by_Smeasure(bar=0.95, loader_comp=loader_comp, bar_comp=0.2)
        dir_good_ones = os.path.join(root_dir_good_ones, dataset)
        os.makedirs(dir_good_ones, exist_ok=True)
        print('have good_ones {}'.format(len(good_ones)))
        for good_one, good_one_comp, good_one_gt in zip(good_ones, good_ones_comp, good_ones_gt):
            dir_category = os.path.join(dir_good_ones, good_one.split('/')[-2])
            os.makedirs(dir_category, exist_ok=True)
            save_path = os.path.join(dir_category, good_one.split('/')[-1])
            sal_map = cv2.imread(good_one)
            sal_map_gt = cv2.imread(good_one_gt)
            sal_map_comp = cv2.imread(good_one_comp)
            image_path = good_one_gt.replace('/gts', '/images').replace('.png', '.jpg')
            image = cv2.imread(image_path)
            cv2.imwrite(save_path, sal_map)
            split_line = np.zeros((sal_map.shape[0], 10, 3)).astype(sal_map.dtype) + 127
            comp = cv2.hconcat([image, split_line, sal_map_gt, split_line, sal_map, split_line, sal_map_comp])
            save_path_comp = ''.join((save_path[:-4], '_comp', save_path[-4:]))
            cv2.imwrite(save_path_comp, comp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='CoCA+CoSOD3k+CoSal2015')
    parser.add_argument('--gt_dir', type=str, default='/root/datasets/sod/gts', help='GT')
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)