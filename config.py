from torch.nn.modules import loss


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
        self.criterion_sal = ['bce', 'iou', 'mse'][0]
        # ACM, GCM
        losses = ['sal', 'cls', 'contrast', 'cls_mask']
        self.loss = losses[:]
        if not self.GAM and 'contrast' in self.loss:
            self.loss.remove('contrast')
        self.loss_sal_last_layers = 1       # used to be last 4 layers
        self.lambda_sal = 100.
        self.lambda_cls = 3.
        self.lambda_contrast = 250.
        self.lambda_cls_mask = 10.

        self.decay_step_size = 30

        self.val_last = 30
