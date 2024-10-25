import json
import multiprocessing as mp
import os
from queue import Empty

import torch as t
from tqdm import tqdm

import wandb

from .dictionary import AutoEncoder
from .evaluation import evaluate
from .trainers.standard import StandardTrainer


def save_checkpoint(wandb_run, model_path, config_path, name, step):
    # Create and log artifact
    artifact = wandb.Artifact(
        name=name,
        type="model",
        description=f"Model checkpoint at step {step}",
    )
    artifact.add_file(model_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    print(f"Model and config saved as artifact at step {step}")


def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            if isinstance(log, dict) and log.get("artifact", False):
                # Handle artifact saving
                artifact_data = log["artifact_data"]
                save_checkpoint(wandb_run=wandb.run, **artifact_data)
            else:
                wandb.log(log)
        except Empty:
            continue
    wandb.finish()


def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list=[],
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
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            # log parameters from training
            log.update({f"{k}": v for k, v in losslog.items()})
            log[f"l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                log[f"{name}"] = value

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
    transcoder=False,
    run_cfg={},
):
    """
    Train SAEs using the given trainers and save them as wandb artifacts
    """
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

        # logging
        if log_steps is not None and step % log_steps == 0:
            log_stats(
                trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues
            )

        # saving
        if save_steps is not None and step % save_steps == 0:
            for dir, trainer, log_queue in zip(save_dirs, trainers, log_queues):
                if dir is not None:
                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))
                    save_path = os.path.join(dir, "checkpoints", f"ae_{step}.pt")
                    t.save(
                        trainer.ae.state_dict(),
                        save_path,
                    )
                    config_path = os.path.join(dir, "config.json")
                # Send message to wandb process to save artifact
                if use_wandb:
                    # Prepare artifact data
                    artifact_data = {
                        "model_path": save_path,
                        "config_path": config_path,
                        "name": f"{trainer.config.get('wandb_name', 'trainer')}_{step}",
                        "step": step,
                    }
                    log_queue.put({"artifact": True, "artifact_data": artifact_data})

        # training
        for trainer in trainers:
            trainer.update(step, act)

    # save final SAEs
    for save_dir, trainer, log_queue in zip(save_dirs, trainers, log_queues):
        if save_dir is not None:
            save_path = os.path.join(save_dir, "ae.pt")
            t.save(trainer.ae.state_dict(), save_path)
            config_path = os.path.join(save_dir, "config.json")
            if use_wandb:
                artifact_data = {
                    "model_path": save_path,
                    "config_path": config_path,
                    "name": f"{trainer.config.get('wandb_name', 'trainer')}_final",
                    "step": 'final',
                }
                log_queue.put({"artifact": True, "artifact_data": artifact_data})

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()
