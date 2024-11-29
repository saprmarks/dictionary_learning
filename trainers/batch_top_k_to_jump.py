import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
from collections import namedtuple
from contextlib import contextmanager

from ..dictionary import Dictionary
from ..trainers.trainer import SAETrainer


class BatchTopKToJumpSAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k

        self.train_mode = False
        self.store_thresholds = False

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.bias.data.zero_()
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T
        self.set_decoder_norm_to_unit_norm()
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        # Initialize running statistics
        self.register_buffer("running_thresholds", t.zeros(dict_size))
        self.register_buffer("threshold_count", t.zeros(1))

    @contextmanager
    def training_mode(self, store_thresholds: bool = False):
        """Context manager for temporarily enabling training mode.

        Args:
            store_thresholds: If True, updates running thresholds during training
        """
        old_mode = self.train_mode
        old_thresholds = self.store_thresholds
        self.train_mode = True
        self.store_thresholds = store_thresholds
        try:
            yield
        finally:
            self.train_mode = old_mode
            self.store_thresholds = old_thresholds

    def encode(self, x: t.Tensor, return_active: bool = False):
        if self.train_mode:
            return self._encode_train(x, return_active)
        else:
            return self._encode_inference(x)

    def _encode_train(self, x: t.Tensor, return_active: bool = False):
        """Used during training - applies BatchTopK and updates thresholds"""
        pre_acts = self.encoder(x - self.b_dec)

        # BatchTopK activation
        post_relu_feat_acts_BF = nn.functional.relu(pre_acts)
        flattened_acts = post_relu_feat_acts_BF.flatten()
        post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = (
            buffer_BF.flatten()
            .scatter(-1, post_topk.indices, post_topk.values)
            .reshape(buffer_BF.shape)
        )

        # Update running mean of thresholds after warmup
        if self.store_thresholds:
            with t.no_grad():
                current_thresholds = t.quantile(
                    post_relu_feat_acts_BF, 1 - self.k / self.dict_size, dim=0
                )
                # Update running mean
                self.threshold_count += 1
                self.running_thresholds = (
                    self.running_thresholds * (self.threshold_count - 1)
                    + current_thresholds
                ) / self.threshold_count

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0
        else:
            return encoded_acts_BF

    def _encode_inference(self, x: t.Tensor):
        """Default encode function - uses JumpReLU style thresholding"""
        pre_acts = self.encoder(x - self.b_dec)
        return pre_acts * (pre_acts > self.running_thresholds).float()

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder.weight.dtype).eps
        norm = t.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.weight.grad is not None
        parallel_component = einops.einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_in d_sae, d_in d_sae -> d_sae",
        )
        self.decoder.weight.grad -= einops.einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_in d_sae -> d_in d_sae",
        )

    @classmethod
    def from_pretrained(
        cls, path, k=None, device=None, **kwargs
    ) -> "BatchTopKToJumpSAE":
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = cls(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class TrainerBatchTopKToJump(SAETrainer):
    def __init__(
        self,
        dict_class=BatchTopKToJumpSAE,
        activation_dim=512,
        dict_size=64 * 512,
        k=8,
        auxk_alpha=1 / 32,
        decay_start=24000,
        steps=30000,
        top_k_aux=512,
        warmup_step_share=0.9,
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name="BatchTopKToJumpSAE",
        submodule_name=None,
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.k = k
        self.warmup_step_share = warmup_step_share
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.ae = dict_class(activation_dim, dict_size, k)

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        scale = dict_size / (2**14)
        self.lr = 2e-4 / scale**0.5
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = top_k_aux

        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        def lr_fn(step):
            if step < decay_start:
                return 1.0
            else:
                return (steps - step) / (steps - decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = ["effective_l0", "dead_features"]
        self.effective_l0 = -1
        self.dead_features = -1

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = t.topk(
                acts[:, dead_features],
                min(self.top_k_aux, dead_features.sum()),
                dim=-1,
            )
            acts_aux = t.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = F.linear(
                acts_aux, self.ae.decoder.weight[:, dead_features]
            )
            l2_loss_aux = (
                self.auxk_alpha
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return t.tensor(0, dtype=x.dtype, device=x.device)

    def loss(self, x, step=None, logging=False):
        store_thresholds = (
            step >= int(self.steps * self.warmup_step_share)
            if step is not None
            else False
        )
        with self.ae.training_mode(store_thresholds=store_thresholds):
            f, active_indices = self.ae.encode(x, return_active=True)
            l0 = (f != 0).float().sum(dim=-1).mean().item()
            x_hat = self.ae.decode(f)

            e = x_hat - x

            self.effective_l0 = self.k

            num_tokens_in_step = x.size(0)
            did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
            did_fire[active_indices] = True
            self.num_tokens_since_fired += num_tokens_in_step
            self.num_tokens_since_fired[did_fire] = 0

            auxk_loss = self.get_auxiliary_loss(x, x_hat, f)

            l2_loss = e.pow(2).sum(dim=-1).mean()
            auxk_loss = auxk_loss.sum(dim=-1).mean()
            loss = l2_loss + self.auxk_alpha * auxk_loss

            if not logging:
                return loss
            else:
                return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                    x,
                    x_hat,
                    f,
                    {
                        "l2_loss": l2_loss.item(),
                        "auxk_loss": auxk_loss.item(),
                        "loss": loss.item(),
                    },
                )

    def update(self, step, x):
        if step == 0:
            median = self.geometric_median(x)
            self.ae.b_dec.data = median

        self.ae.set_decoder_norm_to_unit_norm()
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.ae.remove_gradient_parallel_to_decoder_directions()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "TrainerBatchTopK",
            "dict_class": "BatchTopKToJumpSAE",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break

        return guess
