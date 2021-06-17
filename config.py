class Config():
    def __init__(self) -> None:
        self.bb = ['vgg16', 'resnet50'][0]
        self.relation_module = ['GAM', 'ICE', 'NonLocal', 'MHA'][0]
        self.rand_seed = 7
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:3]      # No improvement from crop and pepper
        self.self_supervision = False
        self.label_smoothing = False
        self.freeze = True

        self.decay_step_size = 20

        self.val_last = 29
