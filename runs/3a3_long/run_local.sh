#!/bin/bash
# This script runs the experiment locally

python -m src.main_lm5 \
"--config-path=../configs/presets" \
"--config-dir=configs" \
"trainer.non_inst.scheduler=cosine" \
"--config-name=3a3_m_tiny" \
'hydra.run.dir=./runs/3a3_long/local/${now:%Y-%m-%d}/${now:%H-%M-%S}/hydra' \
'hydra.sweep.dir=./runs/3a3_long/local/${now:%Y-%m-%d}/${now:%H-%M-%S}/hydra'


python -m src.main_lm5 \
"--config-path=../configs/presets" \
"--config-dir=configs" \
"trainer.non_inst.scheduler=cosine" \
"--config-name=3a3_t_tiny" \
'hydra.run.dir=./runs/3a3_long/local/${now:%Y-%m-%d}/${now:%H-%M-%S}/hydra' \
'hydra.sweep.dir=./runs/3a3_long/local/${now:%Y-%m-%d}/${now:%H-%M-%S}/hydra'


