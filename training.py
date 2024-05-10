"""
Training dictionaries
"""

import torch as t
from .dictionary import AutoEncoder
import os
from tqdm import tqdm
from .trainers.standard import StandardTrainer
import wandb
import json
# from .evaluation import evaluate

def trainSAE(
        data, 
        trainer_configs = [
            {
                'trainer' : StandardTrainer,
                'dict_class' : AutoEncoder,
                'activation_dim' : 512,
                'dictionary_size' : 64*512,
                'lr' : 1e-3,
                'l1_penalty' : 1e-1,
                'warmup_steps' : 1000,
                'resample_steps' : None,
                'seed' : None,
                'wandb_name' : 'StandardTrainer',
            }
        ],
        steps=None,
        save_steps=None,
        save_dir=None, # use {run} to refer to wandb run
        log_steps=None,
        activations_split_by_head=False, # set to true if data is shape [batch, pos, num_head, head_dim/resid_dim]
):
    """
    Train SAEs using the given trainers
    """

    trainers = []
    for config in trainer_configs:
        trainer = config['trainer']
        del config['trainer']
        trainers.append(
            trainer(
                **config
            )
        )

    if log_steps is not None:
        wandb.init(
            entity="sae-training",
            project="sae-training",
            config={f'{trainer.config["wandb_name"]}-{i}' : trainer.config for i, trainer in enumerate(trainers)}
        )
        # process save_dir in light of run name
        if save_dir is not None:
            save_dir = save_dir.format(run=wandb.run.name)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [os.path.join(save_dir, f"trainer{i}") for i in range(len(trainer_configs))]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {'trainer' : trainer.config}
            try:
                config['buffer'] = data.config
            except: pass
            with open(os.path.join(dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]
    
    for step, x in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break
        
        # logging
        if log_steps is not None and step % log_steps == 0:
            log = {}
            with t.no_grad():

                # quick hack to make sure all trainers get the same x
                # TODO make this less hacky
                y = x.clone()
                for i, trainer in enumerate(trainers):
                    x = y.clone()
                    if activations_split_by_head: # x.shape: [batch, pos, n_heads, d_head]
                        x = x[..., i, :] 
                    trainer_name = f'{trainer.config["wandb_name"]}-{i}'
                    x, x_hat, f, losslog = trainer.loss(x, step=step, logging=True)
                    log.update({f'{trainer_name}/{k}' : v for k, v in losslog.items()})

                    # L0
                    log[f'{trainer_name}/l0'] = (f != 0).float().sum(dim=-1).mean().item()

                    # fraction of variance explained
                    total_variance = t.var(x, dim=0).sum()
                    residual_variance = t.var(x - x_hat, dim=0).sum()
                    frac_variance_explained = (1 - residual_variance / total_variance)
                    log[f'{trainer_name}/frac_variance_explained'] = frac_variance_explained.item()

                    # log parameters from training 
                    trainer_log = trainer.get_logging_parameters()
                    for name, value in trainer_log.items():
                        log[f'{trainer_name}/{name}'] = value

                    # TODO get this to work
                    # metrics = evaluate(
                    #     trainer.ae, 
                    #     data, 
                    #     device=trainer.device
                    # )
                    # log.update(
                    #     {f'trainer{i}/{k}' : v for k, v in metrics.items()}
                    # )
            wandb.log(log, step=step)

        # saving
        if save_steps is not None and step % save_steps == 0:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is not None:
                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))
                    t.save(
                        trainer.ae.state_dict(), 
                        os.path.join(dir, "checkpoints", f"ae_{step}.pt")
                        )
                    
        # training
        for trainer in trainers:
            trainer.update(step, x)
    
    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            t.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))