class SAETrainer:
    """
    Generic class for implementing SAE training algorithms
    """
    def __init__(self, ae):
        self.ae = ae
        self.logging_parameters = []

    def update(self, 
               step, # index of step in training
               activations, # of shape [batch_size, d_submodule]
        ):
        pass # implemented by subclasses

    def get_logging_parameters(self):
        stats = {}
        for param in self.logging_parameters:
            if hasattr(self, param):
                stats[param] = getattr(self, param)
        return stats