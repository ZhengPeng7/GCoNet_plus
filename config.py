class Config():
    def __init__(self) -> None:
        self.relation_module = ['NonLocal', 'GAM', 'MHA', 'ICE'][0]
        self.val_last = 15
        self.rand_seed = 7
        self.preproc_methods = ['flip', 'crop', 'rotate', 'enhance', 'pepper'][100:]
