#!/bin/bash

python3 gpt2_sae_trainer.py \
    --save_dir /share/u/can/shift_eval/train_saes/trained_saes/gpt2_jumpConst_sweep0808 \
    --layer 8 \
    # --no_wandb_logging \
    # --dry_run \