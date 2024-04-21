class SAETrainer:
    """
    Generic class for implementing SAE training algorithms
    """
    def __init__(self, ae):
        self.ae = ae

    def update(self, 
               step, # index of step in training
               activations, # of shape [batch_size, d_submodule]
        ):
        pass # implemented by subclasses