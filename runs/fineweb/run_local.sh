#!/bin/bash
# This script runs the experiment locally

python -m src.main_lm5 \
"--config-path=../configs/presets" \
"--config-dir=configs" \
"--config-name=fineweb_test" \
"model/size@model.config=large" \
"+model.config.n_positions=2048" \
"dataset.non_inst.pre_processor.map_args.num_proc=20" \
"dataset.non_inst.pre_processor.map_args.batch_size=1024" \
"dataset.name=sample-10BT" \
"trainer.non_inst.optimizer.lr=0.0003" \
"trainer.args.num_train_epochs=0.1" \
"trainer.callbacks.0.early_stopping_patience=5" \
"trainer.callbacks.1.max_seconds=561600" \
"trainer.args.per_device_train_batch_size=2" \
"trainer.args.per_device_eval_batch_size=2" \
"trainer.args.gradient_accumulation_steps=32" \
"trainer.args.eval_strategy=steps" \
"trainer.args.save_strategy=steps" \
"+trainer.args.save_steps=0.1" \
"+trainer.args.eval_steps=0.1" \
"trainer.args.eval_on_start=False" \
"dataset.non_inst.pre_processor.train2val=0.0001" \
"env.output_dir=/data2/users/gshin22/fineweb_faistos/gpt2_vanilla_l" \
"model=gpt2_vanilla" \
'hydra.run.dir=./runs/fineweb_faistos/local/${now:%Y-%m-%d}/${now:%H-%M-%S}/hydra' \
'hydra.sweep.dir=./runs/fineweb_faistos/local/${now:%Y-%m-%d}/${now:%H-%M-%S}/hydra'


python -m src.main_lm5 \
"--config-path=../configs/presets" \
"--config-dir=configs" \
"--config-name=fineweb_test" \
"model/size@model.config=large" \
"+model.config.n_positions=2048" \
"dataset.non_inst.pre_processor.map_args.num_proc=20" \
"dataset.non_inst.pre_processor.map_args.batch_size=1024" \
"dataset.name=sample-10BT" \
"trainer.non_inst.optimizer.lr=0.0003" \
"trainer.args.num_train_epochs=0.1" \
"trainer.callbacks.0.early_stopping_patience=5" \
"trainer.callbacks.1.max_seconds=561600" \
"trainer.args.per_device_train_batch_size=2" \
"trainer.args.per_device_eval_batch_size=2" \
"trainer.args.gradient_accumulation_steps=32" \
"trainer.args.eval_strategy=steps" \
"trainer.args.save_strategy=steps" \
"+trainer.args.save_steps=0.1" \
"+trainer.args.eval_steps=0.1" \
"trainer.args.eval_on_start=False" \
"dataset.non_inst.pre_processor.train2val=0.0001" \
"env.output_dir=/data2/users/gshin22/fineweb_faistos/gpt2_muse_l" \
"model=gpt2_muse" \
'hydra.run.dir=./runs/fineweb_faistos/local/${now:%Y-%m-%d}/${now:%H-%M-%S}/hydra' \
'hydra.sweep.dir=./runs/fineweb_faistos/local/${now:%Y-%m-%d}/${now:%H-%M-%S}/hydra'


