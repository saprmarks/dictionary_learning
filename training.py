"""
Training dictionaries
"""

import json
import multiprocessing as mp
import os
from queue import Empty
from collections import defaultdict
import torch as t
from tqdm import tqdm

import wandb

from .dictionary import AutoEncoder
from .evaluation import evaluate
from .trainers.standard import StandardTrainer
from .trainers.crosscoder import CrossCoderTrainer

def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            step = log.pop("step")
            wandb.log(log, step=step)
        except Empty:
            continue
    wandb.finish()

def get_stats(
    trainer,
    act: t.Tensor,
    deads_sum: bool=True,
):
    with t.no_grad():
        act, act_hat, f, losslog = trainer.loss(act, step=0, logging=True, return_deads=True)

    # L0
    l0 = (f != 0).float().detach().cpu().sum(dim=-1).mean().item()

    out = {
        "l0": l0,
        **{f"{k}": v for k, v in losslog.items() if k != "deads"},
    }
    if losslog["deads"] is not None:
        total_feats = losslog["deads"].shape[0]
        out["frac_deads"] = losslog["deads"].sum().item() / total_feats if deads_sum else losslog["deads"]

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
            residual_variance_per_layer.append(t.var(act[:, l, :] - act_hat[:, l, :], dim=0).cpu().sum())
            out[f"cl{l}_frac_variance_explained"] = 1 - residual_variance_per_layer[l] / total_variance_per_layer[l]
        total_variance = sum(total_variance_per_layer)
        residual_variance = sum(residual_variance_per_layer)
        frac_variance_explained = 1 - residual_variance / total_variance

        out["frac_variance_explained"] = frac_variance_explained.item()

    return out

def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list=[],
    stage: str="train",
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                stats = get_stats(trainer, act)
                log.update({f"{stage}/{k}": v for k, v in stats.items()})

            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            # log parameters from training
            log["step"] = step
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                log[f"{stage}/{name}"] = value

            if log_queues:
                log_queues[i].put(log)

@t.no_grad()
def run_validation(
    trainers,
    validation_data,
    log_queues: list=[],
    step: int=None,
):
    for i, trainer in enumerate(trainers):
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
                    frac_variance_explained_per_layer[l].append(stats[f"cl{l}_frac_variance_explained"])


        log = {}
        log["val/frac_deads"] = t.stack(deads).all(dim=0).float().mean().item()
        log["val/l0"] = t.tensor(l0).mean().item()
        log["val/frac_variance_explained"] = t.tensor(frac_variance_explained).mean()
        if isinstance(trainer, CrossCoderTrainer):
            for l in range(act.shape[1]):
                log[f"val/cl{l}_frac_variance_explained"] = t.tensor(frac_variance_explained_per_layer[l]).mean()
        if step is not None:
            log["step"] = step
        if log_queues:
            log_queues[i].put(log)

def trainSAE(
    data,
    trainer_configs,
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
    Train SAEs using the given trainers
    """
    assert not(validation_data is None and validate_every_n_steps is not None), "Must provide validation data if validate_every_n_steps is not None"

    trainers = []
    for config in trainer_configs:
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    wandb_processes = []
    log_queues = []

    if use_wandb:
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except:
                pass
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    for step, act in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break
        act = act.to(trainers[0].device)

        # logging
        if log_steps is not None and step % log_steps == 0 and step != 0:
            log_stats(
                trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues
            )

        # saving
        if save_steps is not None and step % save_steps == 0:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is not None:
                    os.makedirs(os.path.join(dir, trainer.config["wandb_name"].lower()), exist_ok=True)
                    t.save(
                        trainer.ae.state_dict() if not trainer_configs[i]["compile"] else trainer.ae._orig_mod.state_dict(),
                        os.path.join(dir, trainer.config["wandb_name"].lower(), f"ae_{step}.pt"),
                    )

        # training
        for trainer in trainers:
            trainer.update(step, act)

        if validate_every_n_steps is not None and step % validate_every_n_steps == 0 and step != 0:
            print(f"Validating at step {step}")
            run_validation(trainers, validation_data, log_queues, step=step)

    # Final validation
    print(f"Validating at step {step}")
    run_validation(trainers, validation_data, log_queues, step=step)

    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            os.makedirs(os.path.join(save_dir, trainer.config["wandb_name"].lower()), exist_ok=True)
            t.save(trainer.ae.state_dict() if not trainer_configs[i]["compile"] else trainer.ae._orig_mod.state_dict(), os.path.join(save_dir, trainer.config["wandb_name"].lower(), f"ae_final.pt"))

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()
