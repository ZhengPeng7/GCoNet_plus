class Config():
    def __init__(self) -> None:
        self.bb = ['vgg16', 'resnet50'][0]
        self.relation_module = ['GAM', 'ICE', 'NonLocal', 'MHA'][0]
        self.rand_seed = 7
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]
        self.self_supervision = False
        self.label_smoothing = False
        self.freeze = True

        self.validation = True

        # Components
        # GAM
        self.GAM = True

        # Loss
        self.criterion_sal = ['bce', 'iou', 'mse'][0]
        # ACM, GCM
        losses = ['sal', 'cls', 'contrast', 'cls_mask']
        self.loss = losses[:]
        if not self.GAM and 'contrast' in self.loss:
            self.loss.remove('contrast')
        if self.criterion_sal == 'bce':
            self.lambda_sal = 25.
        elif self.criterion_sal == 'iou':
            self.lambda_sal = 1.
        elif self.criterion_sal == 'mse':
            self.lambda_sal = 125.

        self.loss_sal_last_layers = 1       # used to be last 4 layers
        self.loss_cls_mask_last_layers = 4       # used to be last 4 layers
        self.activation_out = 'sigmoid'

        self.loss_sal_ratio_by_last_layers = 4 / self.loss_sal_last_layers
        self.loss_cls_mask_ratio_by_last_layers = 4 / self.loss_cls_mask_last_layers
        self.lambda_sal *= self.loss_sal_ratio_by_last_layers
        self.lambda_cls = 3.
        self.lambda_contrast = 250.
        self.lambda_cls_mask = 2.5 * self.loss_cls_mask_ratio_by_last_layers

        # Performance of GCoNet
        self.measures = {
            'Emax': {'CoCA': 0.760, 'CoSOD3k': 0.860, 'CoSal2015': 0.887},
            'Smeasure': {'CoCA': 0.673, 'CoSOD3k': 0.802, 'CoSal2015': 0.845},
            'Fmax': {'CoCA': 0.544, 'CoSOD3k': 0.777, 'CoSal2015': 0.847},
        }

        self.decay_step_size = 300
        self.val_last = 50
