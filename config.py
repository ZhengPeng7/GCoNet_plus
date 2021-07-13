class Config():
    def __init__(self) -> None:
        self.bb = ['vgg16', 'vgg16bn', 'resnet50'][1]
        self.use_bn = 'bn' in self.bb or 'resnet' in self.bb
        self.relation_module = ['GAM', 'ICE', 'NonLocal', 'MHA'][0]
        self.rand_seed = 7
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]
        self.self_supervision = False
        self.label_smoothing = False
        self.freeze = True

        self.validation = False

        # Components
        # GAM
        self.GAM = True
        self.refine = [0, 1, 4][0]         # 0 -- no refinement, 1 -- only output mask for refinement, 4 -- but also raw input.
        if self.refine:
            self.batch_size = 16
        else:
            if self.bb != 'vgg16':
                self.batch_size = 32
            else:
                self.batch_size = 48
        self.lr = 1e-4 * (self.batch_size / 16)
        self.split_mask = True
        self.cls_mask_operation = ['x', '+', 'c'][0]
        self.db_mask = False and self.split_mask
        self.db_output_decoder = False
        self.db_output_refiner = False and self.refine
        self.db_k = 300

        # Loss
        # ACM, GCM
        losses = ['sal', 'cls', 'contrast', 'cls_mask']
        self.loss = losses[:]
        if not self.GAM and 'contrast' in self.loss:
            self.loss.remove('contrast')
        self.lambdas_sal_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 25
            'bce': 30,
            'iou': 0.5,
            'ssim': 1 * 0,
            'mse': 150 * 0,
            'reg': 50 * 0,
            'triplet': 10 * ('cls' in self.loss),
        }
        self.triplet = ['_x5', 'mask'][:1]
        self.lambdas_sal_others = {
            'bce': 0,
            'iou': 0.5,
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

        self.lambda_adv = 0.        # turn to 0 to avoid adv training

        # Performance of GCoNet
        self.measures = {
            'Emax': {'CoCA': 0.760, 'CoSOD3k': 0.860, 'CoSal2015': 0.887},
            'Smeasure': {'CoCA': 0.673, 'CoSOD3k': 0.802, 'CoSal2015': 0.845},
            'Fmax': {'CoCA': 0.544, 'CoSOD3k': 0.777, 'CoSal2015': 0.847},
        }

        self.decay_step_size = 300
        self.val_last = 60
