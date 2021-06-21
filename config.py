class Config():
    def __init__(self) -> None:
        self.bb = ['vgg16', 'resnet50'][0]
        self.relation_module = ['GAM', 'ICE', 'NonLocal', 'MHA'][0]
        self.rand_seed = 7
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]      # No improvement from crop and pepper
        self.self_supervision = False
        self.label_smoothing = False
        self.freeze = True

        # model
        # baseline model doesn't contain modules for loss_cls, loss_x5
        self.model = ['bsl', 'GCoNet', 'GCoNet_ext'][1]

        # loss
        self.loss = ['sal', 'cls', 'x5']
        self.loss_sal_last_layers = 1       # used to be last 4 layers
        self.lambda_dsloss = 300

        self.decay_step_size = 30

        self.val_last = 29
