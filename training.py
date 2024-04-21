"""
Training dictionaries
"""

import torch as t
from .dictionary import AutoEncoder
import os
from tqdm import tqdm
from .trainers.standard import StandardTrainer
import wandb
from .evaluation import sae_loss

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
    
    wandb.init(
        project="dictionary_learning",
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
        
        for i, trainer in enumerate(trainers):
            # logging
            if log_steps is not None and step % log_steps == 0:
                with t.no_grad():
                    l2_loss, l1_loss, l0 = sae_loss(trainer.ae, activations)
                wandb.log(
                    {
                        f"trainer{i}/l2_loss" : l2_loss,
                        f"trainer{i}/l1_loss" : l1_loss,
                        f"trainer{i}/l0" : l0,
                    }
                )

            # saving
            if save_steps is not None and save_dirs[i] is not None and step % save_steps == 0:
                if not os.path.exists(os.path.join(save_dirs[i], "checkpoints")):
                    os.mkdir(os.path.join(save_dirs[i], "checkpoints"))
                t.save(
                    trainer.ae.state_dict(), 
                    os.path.join(save_dirs[i], "checkpoints", f"ae_{step}.pt")
                    )
                
            trainer.update(step, activations)
    
    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if save_dir is not None:
            t.save(trainer.ae.state_dict(), os.path.join(save_dir, "ae.pt"))