import os


class Config():
    def __init__(self) -> None:
        # Backbone
        self.bb = ['vgg16', 'vgg16bn', 'resnet50'][1]
        # BN
        self.use_bn = 'bn' in self.bb or 'resnet' in self.bb
        # Augmentation
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]

        # Mask
        losses = ['sal', 'cls', 'contrast', 'cls_mask']
        self.loss = losses[:]
        self.split_mask = True and 'cls_mask' in self.loss
        self.cls_mask_operation = ['x', '+', 'c'][0]
        # Loss + Triplet Loss
        self.lambdas_sal_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 30 * 1,          # high performance
            'iou': 0.5 * 1,         # 0 / 255
            'ssim': 1 * 0,          # help contours
            'mse': 150 * 0,         # can smooth the saliency map
            'reg': 100 * 0,
            'triplet': 3 * 0 * ('cls' in self.loss),
        }

        # DB
        self.db_output_decoder = True
        self.db_k = 300
        self.db_k_alpha = 1
        self.db_mask = False and self.split_mask

        # Triplet Loss
        self.triplet = ['_x5', 'mask'][:1]
        self.triplet_loss_margin = 0.1
        # Adv
        self.lambda_adv = 0.        # turn to 0 to avoid adv training

        # Refiner
        self.refine = [0, 1, 4][0]         # 0 -- no refinement, 1 -- only output mask for refinement, 4 -- but also raw input.
        if self.refine:
            self.batch_size = 16
        else:
            if self.bb != 'vgg16':
                self.batch_size = 32
            else:
                self.batch_size = 48
        self.db_output_refiner = False and self.refine

        # Intermediate Layers
        self.lambdas_sal_others = {
            'bce': 0,
            'iou': 0.,
            'ssim': 0,
            'mse': 0,
            'reg': 0,
            'triplet': 0,
        }
        self.output_number = 1
        self.loss_sal_layers = 4              # used to be last 4 layers
        self.loss_cls_mask_last_layers = 1         # used to be last 4 layers
        if 'keep in range':
            self.loss_sal_layers = min(self.output_number, self.loss_sal_layers)
            self.loss_cls_mask_last_layers = min(self.output_number, self.loss_cls_mask_last_layers)
            self.output_number = min(self.output_number, max(self.loss_sal_layers, self.loss_cls_mask_last_layers))
            if self.output_number == 1:
                for cri in self.lambdas_sal_others:
                    self.lambdas_sal_others[cri] = 0
        self.conv_after_itp = False
        self.complex_lateral_connection = False

        # to control the quantitive level of each single loss by number of output branches.
        self.loss_cls_mask_ratio_by_last_layers = 4 / self.loss_cls_mask_last_layers
        for loss_sal in self.lambdas_sal_last.keys():
            loss_sal_ratio_by_last_layers = 4 / (int(bool(self.lambdas_sal_others[loss_sal])) * (self.loss_sal_layers - 1) + 1)
            self.lambdas_sal_last[loss_sal] *= loss_sal_ratio_by_last_layers
            self.lambdas_sal_others[loss_sal] *= loss_sal_ratio_by_last_layers
        self.lambda_cls_mask = 2.5 * self.loss_cls_mask_ratio_by_last_layers
        self.lambda_cls = 3.
        self.lambda_contrast = 250.

        # Performance of GCoNet
        self.val_measures = {
            'Emax': {'CoCA': 0.760, 'CoSOD3k': 0.860, 'CoSal2015': 0.887},
            'Smeasure': {'CoCA': 0.673, 'CoSOD3k': 0.802, 'CoSal2015': 0.845},
            'Fmax': {'CoCA': 0.544, 'CoSOD3k': 0.777, 'CoSal2015': 0.847},
        }

        # others
        self.GAM = True
        if not self.GAM and 'contrast' in self.loss:
            self.loss.remove('contrast')
        self.lr = 1e-4 * (self.batch_size / 16)
        self.relation_module = ['GAM', 'ICE', 'NonLocal', 'MHA'][0]
        self.self_supervision = False
        self.label_smoothing = False
        self.freeze = True

        self.validation = False
        self.decay_step_size = 3000
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'gco' in f and '.sh' in f] + [os.path.join('..', f) for f in os.listdir('..') if 'gco' in f and '.sh' in f]
        with open(run_sh_file[0], 'r') as f:
            self.val_last = int([l.strip() for l in f.readlines() if 'val_last=' in l][0].split('=')[-1])
