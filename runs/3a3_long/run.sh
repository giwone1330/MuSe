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
tar -xzf /staging/gshin22/hf_secure/hf_secure_260316022129.tar.gz -C .

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
--config-name="${5}" \
trainer.non_inst.scheduler="cosine" \
hydra.run.dir="./runs/3a3_long/local/hydra" \
hydra.sweep.dir="./runs/3a3_long/local/hydra" \
env.chtc.cluster="${3}" \
env.chtc.process="${4}" \
env.chtc.model_config="${1}" \

echo "End of python file"
echo
echo $PWD
./scripts/create_file_tree.sh .