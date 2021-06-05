class Config():
    def __init__(self) -> None:
        self.relation_module = ['NonLocal', 'GAM', 'MHA', 'ICE'][1]
        self.val_last = 10
        self.rand_seed = 0
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'pepper', 'crop'][:3]
        self.self_supervision = False
        self.label_smoothing = False
