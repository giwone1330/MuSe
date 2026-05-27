#!/bin/bash
conda init
source /opt/conda/etc/profile.d/conda.sh
conda env list
if conda activate myvenv 2>/dev/null; then
    echo "Using conda env 'myvenv'"
else
    echo "Failed to activate 'myvenv', running rest of script..."
fi
# Unpack the data
tar -xzf /staging/gshin22/hf_secure/hf_secure.tar.gz -C .

# Check and unpack the data
if [ -f "/staging/gshin22/hf_secure/data.tar.gz" ]; then
    echo "Found /staging/gshin22/hf_secure/data.tar.gz. Extracting..."
    tar -xzf /staging/gshin22/hf_secure/data.tar.gz -C .
    echo "Extraction done."
else
    echo "/staging/gshin22/hf_secure/data.tar.gz does not exist. Skipping data extraction."
fi

model_config=${1}
number=${2}
Cluster=${3}
Process=${4}

cd hf_secure
python -m src.main_lm5 \
--config-path="../configs/presets" \
--config-dir="configs" \
--config-name="fineweb_test" \
model="${5}" \
model/size@model.config="large" \
+model.config.n_positions="2048" \
dataset.non_inst.pre_processor.map_args.num_proc="40" \
dataset.non_inst.pre_processor.map_args.batch_size="1024" \
dataset.name="sample-10BT" \
trainer.non_inst.optimizer.lr="0.0003" \
trainer.args.num_train_epochs="0.1" \
trainer.callbacks.0.early_stopping_patience="5" \
trainer.callbacks.1.max_seconds="561600" \
trainer.args.per_device_train_batch_size="2" \
trainer.args.per_device_eval_batch_size="2" \
trainer.args.gradient_accumulation_steps="32" \
trainer.args.eval_strategy="steps" \
trainer.args.save_strategy="steps" \
+trainer.args.save_steps="0.1" \
+trainer.args.eval_steps="0.1" \
trainer.args.eval_on_start="False" \
dataset.non_inst.pre_processor.train2val="0.0001" \
env.output_dir="/data2/users/gshin22/fineweb_faistos/temp" \
hydra.run.dir="./runs/fineweb_faistos/local/hydra" \
hydra.sweep.dir="./runs/fineweb_faistos/local/hydra" \
env.chtc.cluster="${3}" \
env.chtc.process="${4}" \
env.chtc.model_config="${1}" \

echo "End of python file"
echo
echo $PWD
./scripts/create_file_tree.sh .