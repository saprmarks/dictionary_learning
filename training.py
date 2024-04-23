"""
Training dictionaries
"""

import torch as t
from .dictionary import AutoEncoder
import os
from tqdm import tqdm
from .trainers.standard import StandardTrainer
import wandb
# from .evaluation import evaluate

def _set_seeds(seed):
    if seed is not None:
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)

def trainSAE(
        data, 
        activation_dim,
        dictionary_size,
        trainer_configs = [
            {
                'trainer' : StandardTrainer,
                'lr' : 1e-3,
                'l1_penalty' : 1e-1,
                'warmup_steps' : 1000,
                'resample_steps' : None,
            }
        ],
        steps=None,
        save_steps=None,
        save_dirs=[None],
        log_steps=None,
        seed=None,
):
    """
    Train SAEs using the given trainers
    """

    # make save_dirs if they don't already exist
    for save_dir in save_dirs:
        if save_dir is not None and not os.path.exists(save_dir):
            os.mkdir(save_dir)
    if log_steps is not None:
        wandb.init(
            entity="sae-training",
            project="sae-training",
            config={f'trainer{i}' : config for i, config in enumerate(trainer_configs)}
        )

    trainers = []
    for config in trainer_configs:
        trainer = config['trainer']
        del config['trainer']
        _set_seeds(seed)
        trainers.append(
            trainer(
                AutoEncoder(activation_dim, dictionary_size),
                **config
            )
        )
    
    for step, activations in enumerate(tqdm(data, total=steps)):
        if steps is not None and step >= steps:
            break
        
        # logging
        if log_steps is not None and step % log_steps == 0:
            log = {}
            for i, trainer in enumerate(trainers):
                x_hat, f = trainer.ae(activations, output_features=True)
                l2_loss = t.linalg.norm(activations - x_hat, dim=-1).mean()
                l1_loss = f.norm(p=1, dim=-1).mean()
                l0 = (f != 0).float().sum(dim=-1).mean()
                frac_alive = (f != 0).float().mean(dim=-1).mean()

                #compute variance explained
                total_variance = t.var(activations, dim=0).sum()
                residual_variance = t.var(activations - x_hat, dim=0).sum()
                frac_variance_explained = (1 - residual_variance / total_variance)

                log[f'trainer{i}/l2_loss'] = l2_loss.item()
                log[f'trainer{i}/l1_loss'] = l1_loss.item()
                log[f'trainer{i}/l0'] = l0.item()
                log[f'trainer{i}/frac_alive'] = frac_alive.item()
                log[f'trainer{i}/frac_variance_explained'] = frac_variance_explained.item()

                # TODO get this to work
                # metrics = evaluate(
                #     trainer.ae, 
                #     data, 
                #     device=trainer.device
                # )
                # log.update(
                #     {f'trainer{i}/{k}' : v for k, v in metrics.items()}
                # )
            wandb.log(log)

        # saving
        if save_steps is not None and step % save_steps == 0:
            for i, trainer in enumerate(trainers):
                if save_dirs[i] is not None:
                    if not os.path.exists(os.path.join(save_dirs[i], "checkpoints")):
                        os.mkdir(os.path.join(save_dirs[i], "checkpoints"))
                    t.save(
                        trainer.ae.state_dict(), 
                        os.path.join(save_dirs[i], "checkpoints", f"ae_{step}.pt")
                        )
                    
        # training
        for trainer in trainers:
            trainer.update(step, activations)
    
    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            t.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))