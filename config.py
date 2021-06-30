class Config():
    def __init__(self) -> None:
        self.bb = ['vgg16', 'resnet50'][0]
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

        # Loss
        self.lambdas_sal = {
            # not 0 means opening this loss
            'bce': 25 * 1,
            'ssim': 25 * 0,
            'mse': 125 * 0,
            'iou': 1 * 0,
        }
        # ACM, GCM
        losses = ['sal', 'cls', 'contrast', 'cls_mask']
        self.loss = losses[:]
        if not self.GAM and 'contrast' in self.loss:
            self.loss.remove('contrast')

        self.output_number = 1
        self.loss_sal_last_layers = min(self.output_number, 1)              # used to be last 4 layers
        self.loss_cls_mask_last_layers = min(self.output_number, 4)         # used to be last 4 layers
        self.conv_after_itp = False
        self.complex_lateral_connection = False

        # to control the quantitive level of each single loss by number of output branches.
        self.loss_sal_ratio_by_last_layers = 4 / self.loss_sal_last_layers
        self.loss_cls_mask_ratio_by_last_layers = 4 / self.loss_cls_mask_last_layers
        for loss_sal in self.lambdas_sal.keys():
            self.lambdas_sal[loss_sal] *= self.loss_sal_ratio_by_last_layers
        self.lambda_cls_mask = 2.5 * self.loss_cls_mask_ratio_by_last_layers
        self.lambda_cls = 3.
        self.lambda_contrast = 250.

        # Performance of GCoNet
        self.measures = {
            'Emax': {'CoCA': 0.760, 'CoSOD3k': 0.860, 'CoSal2015': 0.887},
            'Smeasure': {'CoCA': 0.673, 'CoSOD3k': 0.802, 'CoSal2015': 0.845},
            'Fmax': {'CoCA': 0.544, 'CoSOD3k': 0.777, 'CoSal2015': 0.847},
        }

        self.decay_step_size = 300
        self.val_last = 30
