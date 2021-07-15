from torch.utils import data
import os
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root, return_predpath=False, return_gtpath=False):
        self.return_predpath = return_predpath
        self.return_gtpath = return_gtpath
        pred_dirs = os.listdir(pred_root)
        label_dirs = os.listdir(label_root)

        dir_name_list = []
        for idir in pred_dirs:
            if idir in label_dirs:
                pred_names = os.listdir(os.path.join(pred_root, idir))
                label_names = os.listdir(os.path.join(label_root, idir))
                for iname in pred_names:
                    if iname in label_names:
                        dir_name_list.append(os.path.join(idir, iname))

        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), dir_name_list))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), dir_name_list))

        self.labels = []
        for p in self.label_path:
            self.labels.append(Image.open(p).convert('L'))


    def __getitem__(self, item):
        predpath = self.image_path[item]
        gtpath = self.label_path[item]
        pred = Image.open(predpath).convert('L')
        gt = self.labels[item]
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        returns = [pred, gt]
        if self.return_predpath:
            returns.append(predpath)
        if self.return_gtpath:
            returns.append(gtpath)
        return returns

    def __len__(self):
        return len(self.image_path)
