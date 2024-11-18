"""
Training dictionaries
"""

import json
import os
from collections import defaultdict
import torch as t
from tqdm import tqdm

import wandb

from .dictionary import AutoEncoder
from .evaluation import evaluate
from .trainers.standard import StandardTrainer
from .trainers.crosscoder import CrossCoderTrainer


def get_stats(
    trainer,
    act: t.Tensor,
    deads_sum: bool = True,
):
    with t.no_grad():
        act, act_hat, f, losslog = trainer.loss(
            act, step=0, logging=True, return_deads=True
        )

    # L0
    l0 = (f != 0).float().detach().cpu().sum(dim=-1).mean().item()

    out = {
        "l0": l0,
        **{f"{k}": v for k, v in losslog.items() if k != "deads"},
    }
    if losslog["deads"] is not None:
        total_feats = losslog["deads"].shape[0]
        out["frac_deads"] = (
            losslog["deads"].sum().item() / total_feats
            if deads_sum
            else losslog["deads"]
        )

    # fraction of variance explained
    if act.dim() == 2:
        # act.shape: [batch, d_model]
        # fraction of variance explained
        total_variance = t.var(act, dim=0).sum()
        residual_variance = t.var(act - act_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
    else:
        # act.shape: [batch, layer, d_model]
        total_variance_per_layer = []
        residual_variance_per_layer = []

        for l in range(act.shape[1]):
            total_variance_per_layer.append(t.var(act[:, l, :], dim=0).cpu().sum())
            residual_variance_per_layer.append(
                t.var(act[:, l, :] - act_hat[:, l, :], dim=0).cpu().sum()
            )
            out[f"cl{l}_frac_variance_explained"] = (
                1 - residual_variance_per_layer[l] / total_variance_per_layer[l]
            )
        total_variance = sum(total_variance_per_layer)
        residual_variance = sum(residual_variance_per_layer)
        frac_variance_explained = 1 - residual_variance / total_variance

        out["frac_variance_explained"] = frac_variance_explained.item()

    return out


def log_stats(
    trainer,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    stage: str = "train",
):
    with t.no_grad():
        log = {}
        if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
            act = act[..., 0, :]
        if not transcoder:
            stats = get_stats(trainer, act)
            log.update({f"{stage}/{k}": v for k, v in stats.items()})
        else:  # transcoder
            x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)
            # L0
            l0 = (f != 0).float().sum(dim=-1).mean().item()
            log[f"{stage}/l0"] = l0

        # log parameters from training
        log["step"] = step
        trainer_log = trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            log[f"{stage}/{name}"] = value

        wandb.log(log, step=step)


@t.no_grad()
def run_validation(
    trainer,
    validation_data,
    step: int = None,
):
    l0 = []
    frac_variance_explained = []
    deads = []
    if isinstance(trainer, CrossCoderTrainer):
        frac_variance_explained_per_layer = defaultdict(list)
    for val_step, act in enumerate(tqdm(validation_data, total=len(validation_data))):
        act = act.to(trainer.device)
        stats = get_stats(trainer, act, deads_sum=False)
        l0.append(stats["l0"])
        deads.append(stats["frac_deads"])
        frac_variance_explained.append(stats["frac_variance_explained"])
        if isinstance(trainer, CrossCoderTrainer):
            for l in range(act.shape[1]):
                frac_variance_explained_per_layer[l].append(
                    stats[f"cl{l}_frac_variance_explained"]
                )

    log = {}
    log["val/frac_deads"] = t.stack(deads).all(dim=0).float().mean().item()
    log["val/l0"] = t.tensor(l0).mean().item()
    log["val/frac_variance_explained"] = t.tensor(frac_variance_explained).mean()
    if isinstance(trainer, CrossCoderTrainer):
        for l in range(act.shape[1]):
            log[f"val/cl{l}_frac_variance_explained"] = t.tensor(
                frac_variance_explained_per_layer[l]
            ).mean()
    if step is not None:
        log["step"] = step
    wandb.log(log, step=step)


def trainSAE(
    data,
    trainer_config,
    use_wandb=False,
    wandb_entity="",
    wandb_project="",
    steps=None,
    save_steps=None,
    save_dir=None,
    log_steps=None,
    activations_split_by_head=False,
    validate_every_n_steps=None,
    validation_data=None,
    transcoder=False,
    run_cfg={},
):
    """
    Train SAE using the given trainer
    """
    assert not (
        validation_data is None and validate_every_n_steps is not None
    ), "Must provide validation data if validate_every_n_steps is not None"

    trainer_class = trainer_config["trainer"]
    del trainer_config["trainer"]
    trainer = trainer_class(**trainer_config)

    wandb_config = trainer.config | run_cfg
    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config=wandb_config,
        name=wandb_config["wandb_name"],
        mode="disabled" if not use_wandb else "online",
    )

    # make save dir, export config
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # save config
        config = {"trainer": trainer.config}
        try:
            config["buffer"] = data.config
        except:
            pass
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break
        act = act.to(trainer.device)

        # logging
        if log_steps is not None and step % log_steps == 0 and step != 0:
            log_stats(trainer, step, act, activations_split_by_head, transcoder)

        # saving
        if save_steps is not None and step % save_steps == 0:
            if save_dir is not None:
                os.makedirs(
                    os.path.join(save_dir, trainer.config["wandb_name"].lower()),
                    exist_ok=True,
                )
                t.save(
                    (
                        trainer.ae.state_dict()
                        if not trainer_config["compile"]
                        else trainer.ae._orig_mod.state_dict()
                    ),
                    os.path.join(
                        save_dir, trainer.config["wandb_name"].lower(), f"ae_{step}.pt"
                    ),
                )

        # training
        trainer.update(step, act)

        if (
            validate_every_n_steps is not None
            and step % validate_every_n_steps == 0
            and step != 0
        ):
            print(f"Validating at step {step}")
            run_validation(trainer, validation_data, step=step)

    try:
        run_validation(trainer, validation_data, step=step)
    except Exception as e:
        print(f"Error during final validation: {str(e)}")

    # save final SAE
    if save_dir is not None:
        os.makedirs(
            os.path.join(save_dir, trainer.config["wandb_name"].lower()), exist_ok=True
        )
        t.save(
            (
                trainer.ae.state_dict()
                if not trainer_config["compile"]
                else trainer.ae._orig_mod.state_dict()
            ),
            os.path.join(
                save_dir, trainer.config["wandb_name"].lower(), f"ae_final.pt"
            ),
        )

    if use_wandb:
        wandb.finish()
