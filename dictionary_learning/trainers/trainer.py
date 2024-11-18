class SAETrainer:
    """
    Generic class for implementing SAE training algorithms
    """

    def __init__(self, seed=None):
        self.seed = seed
        self.logging_parameters = []

    def update(
        self,
        step,  # index of step in training
        activations,  # of shape [batch_size, d_submodule]
    ):
        pass  # implemented by subclasses

    def get_logging_parameters(self):
        stats = {}
        for param in self.logging_parameters:
            if hasattr(self, param):
                stats[param] = getattr(self, param)
            else:
                print(f"Warning: {param} not found in {self}")
        return stats

    @property
    def config(self):
        return {
            "wandb_name": "trainer",
        }
