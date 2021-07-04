class Config():
    def __init__(self) -> None:
        self.bb = ['vgg16', 'vgg16bn', 'resnet50'][1]
        self.relation_module = ['GAM', 'ICE', 'NonLocal', 'MHA'][0]
        self.rand_seed = 7
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]
        self.self_supervision = False
        self.label_smoothing = False
        self.freeze = True
        self.use_bn = 'bn' in self.bb or 'resnet' in self.bb

        self.validation = False

        # Components
        # GAM
        self.GAM = True
        self.refine = [0, 1, 4][2]         # 0 -- no refinement, 1 -- only output mask for refinement, 4 -- but also raw input.
        if self.refine or self.bb != 'vgg16':
            self.batch_size = 32
        else:
            self.batch_size = 48
        self.split_mask = True

        # Loss
        self.lambdas_sal_last = {
            # not 0 means opening this loss
            # original rate -- 25 : 1 : 25 : 125
            'bce': 1,
            'iou': 1,
            'ssim': 1,
            'mse': 0,
        }
        self.lambdas_sal_others = {
            'bce': 0,
            'iou': 0,
            'ssim': 0,
            'mse': 0,
        }
        # ACM, GCM
        losses = ['sal', 'cls', 'contrast', 'cls_mask']
        self.loss = losses[:]
        if not self.GAM and 'contrast' in self.loss:
            self.loss.remove('contrast')

        self.output_number = 1
        self.loss_sal_last_layers = 4              # used to be last 4 layers
        self.loss_cls_mask_last_layers = 1         # used to be last 4 layers
        if 'keep in range':
            self.loss_sal_last_layers, self.loss_cls_mask_last_layers = min(self.output_number, 1), min(self.output_number, 1)
            self.output_number = min(self.output_number, max(self.loss_sal_last_layers, self.loss_cls_mask_last_layers))
            if self.output_number == 1:
                for cri in self.lambdas_sal_others:
                    self.lambdas_sal_others[cri] = 0
        self.conv_after_itp = False
        self.complex_lateral_connection = False

        # to control the quantitive level of each single loss by number of output branches.
        self.loss_sal_ratio_by_last_layers = 4 / self.loss_sal_last_layers
        self.loss_cls_mask_ratio_by_last_layers = 4 / self.loss_cls_mask_last_layers
        for loss_sal in self.lambdas_sal_last.keys():
            self.lambdas_sal_last[loss_sal] *= self.loss_sal_ratio_by_last_layers
        for loss_sal in self.lambdas_sal_others.keys():
            self.lambdas_sal_others[loss_sal] *= self.loss_sal_ratio_by_last_layers
        self.lambda_cls_mask = 2.5 * self.loss_cls_mask_ratio_by_last_layers
        self.lambda_cls = 3.
        self.lambda_contrast = 250.

        self.lambda_adv = 0.        # turn to 0 to avoid adv training

        # Performance of GCoNet
        self.measures = {
            'Emax': {'CoCA': 0.760, 'CoSOD3k': 0.860, 'CoSal2015': 0.887},
            'Smeasure': {'CoCA': 0.673, 'CoSOD3k': 0.802, 'CoSal2015': 0.845},
            'Fmax': {'CoCA': 0.544, 'CoSOD3k': 0.777, 'CoSal2015': 0.847},
        }

        self.decay_step_size = 300
        self.val_last = 30
