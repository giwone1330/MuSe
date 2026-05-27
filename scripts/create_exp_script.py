import yaml
import argparse
import os
from collections import OrderedDict
from itertools import product
import sys
import glob
import json

# # hydra
# import hydra
# from omegaconf import DictConfig, OmegaConf

from pathlib import Path
from typing import Tuple


def get_latest_tarballs(shared_path: str, project_dir: str) -> Tuple[str, str]:
    """
    Returns the filenames of the latest project and data tarballs
    located in /staging/<shared_path>.

    :param shared_path: Shared staging subdirectory
    :param project_dir: Project directory name (prefix for project archive)
    :return: (project_tarball_name, data_tarball_name)
    :raises FileNotFoundError: if either archive is missing
    """
    archive_dir = Path(f"/staging/{shared_path}")

    if not archive_dir.exists():
        print(f"Archive directory not found: {archive_dir}, defaulting to default tarball names")
        return f"{project_dir}.tar.gz", "data.tar.gz"

    # Find latest project tarball
    project_archives = sorted(
        archive_dir.glob(f"{project_dir}_*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    # Find latest data tarball
    data_archives = sorted(
        archive_dir.glob("data_*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )


    if not project_archives:
        raise FileNotFoundError(f"No project tarball found for prefix '{project_dir}_'")
    proj_tar = project_archives[0].name

    if not data_archives:
        data_tar = "data.tar.gz"  # Default name if no data tarball is found
    else:
        data_tar = data_archives[0].name

    return proj_tar, data_tar


def create_queue(input_dict):
    keys = list(input_dict.keys())  # Store headers
    values = input_dict.values()
    combinations = [list(comb) for comb in product(*values)]
    combinations = [[f'{element}'.replace(", ", ",") for element in sublist] for sublist in combinations]
    return keys, combinations  # Prepend headers

def load_config(file_path):
    """Reads a YAML configuration file and returns a dictionary."""
    try:
        with open(os.path.join(file_path, "run.yaml"), "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The folder '{file_path}' does not have run.yaml.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    return {}

def generate_ouput_mappings(output_list, project_name, user_name):

    outputs = []
    transfer_output_files_list = []
    transfer_output_remaps_list = []
    for i, output_file_path in enumerate(output_list):
        output_file_name = os.path.basename(output_file_path)
        output_file_dir = output_file_path.replace(output_file_name, "")
        outputs.append(f"output{i} = $(container_local_output_directory)/{output_file_path}")
        transfer_output_files_list.append(f"$(output{i})")
        transfer_output_remaps_list.append(f"{output_file_name}=/home/{user_name}/$(output_directory)/{output_file_path}")

    output_str = "\n".join(outputs)
    transfer_output_files = ", ".join(transfer_output_files_list)
    transfer_output_files = f"transfer_output_files = {transfer_output_files}"
    transfer_output_remaps = "; ".join(transfer_output_remaps_list)
    transfer_output_remaps = f'transfer_output_remaps = "{transfer_output_remaps}"'

    return f"{output_str}\n\n{transfer_output_files}\n\n{transfer_output_remaps}"

def generate_train_script(config, args):
    """Generates a run.sh script with command-line arguments from config."""
    
    exp_path = args.experiment_path
    project_name = args.project_name
    user_name = args.user_name
    shared_path = args.shared_path

    # get the exact names of the tarballs
    project_tarball, data_tarball = get_latest_tarballs(shared_path, project_name)


    sub_id = OrderedDict() # namely model_config and number
    sub_id["model_config"]="model_config"
    sub_id["number"]="number"
    sub_id["Cluster"]="Cluster"
    sub_id["Process"]="Process"
    sub_queue_args = OrderedDict() # the input to the run.sh file, possibly from run.sub

    # get arguments from submit file
    sub_args = []
    for i, (key, value) in enumerate(sub_id.items()):
        sub_args.append(f"{key}=${{{i+1}}}")

    # Config
    train_py = config.pop("train_py", ["src/main.py"])[0]
    train_py_root = train_py.replace(".py", "")
    train_py_root = train_py_root.replace("/", ".")

    args = []
    args_local = []
    args_local_iter = OrderedDict()
    launch_args_init = []
    # fixed arguments
    # args.append(f'ckpt_path_name="${{model_config}}_${{number}}.pt"')
    # args.append(f'out_dir="out/monet/masking/${{model_config}}"')
    backslash_char = "\\"



    # process run.yaml
    for key, value in config.items():
        # parse the prefix for hydra
        prefix = ""
        if key.startswith("++"):
            prefix = "++"
            key = key[2:]
        elif key[0] in ['+', '~']:
            prefix = key[0]
            key = key[1:]
        
        # parse the arguments
        if len(value)==1:
            if value[0]=="enable_flag":
                args.append(f'{prefix}{key} {backslash_char}\n')
                args_local.append(f'"{prefix}{key} {backslash_char}\n"')
            else:
                if type(value[0]) == type("string"):
                    args.append(f'{prefix}{key}="{value[0]}" {backslash_char}\n')
                    args_local.append(f'"{prefix}{key}={value[0]}" {backslash_char}\n')
                    launch_args_init.append(f"{prefix}{key}={value[0]}")
                else:
                    args.append(f'{prefix}{key}="{value[0]}" {backslash_char}\n')
                    args_local.append(f'"{prefix}{key}={value[0]}" {backslash_char}\n')
                    launch_args_init.append(f"{prefix}{key}={value[0]}")
        else:
            sub_queue_args[f"{key}"] = value
            args_local_iter[f"{prefix}{key}"] = value
            len(sub_queue_args)
            args.append(f'{prefix}{key}="${{{len(sub_queue_args)+len(sub_id)}}}" {backslash_char}\n')

    temp = ''.join(args)

    # model_config=${1}
    # number=${2}
    # Cluster=${3}
    # Process=${4}
    chtc_cluster =  f'env.chtc.cluster="${{3}}" {backslash_char}\n'
    chtc_process =  f'env.chtc.process="${{4}}" {backslash_char}\n'
    chtc_model_config =  f'env.chtc.model_config="${{1}}" {backslash_char}\n'
    command = f"python -m {train_py_root} {backslash_char}\n{temp}"

    # for i, (key, value) in enumerate(sub_queue_args.items()):
    #     sub_args.append(f"{key}=${{{i+1+len(sub_id)}}}")
    
    temp = '\n'.join(sub_args)
    sub_args_remap = f"{temp}\n"
    
    unpack = f"""# Unpack the data
tar -xzf /staging/{user_name}/{project_name}/{project_tarball} -C .
"""
    
    unpack_data = f'''# Check and unpack the data
if [ -f "/staging/{user_name}/{project_name}/{data_tarball}" ]; then
    echo "Found /staging/{user_name}/{project_name}/{data_tarball}. Extracting..."
    tar -xzf /staging/{user_name}/{project_name}/{data_tarball} -C .
    echo "Extraction done."
else
    echo "/staging/{user_name}/{project_name}/{data_tarball} does not exist. Skipping data extraction."
fi
'''
    
    compress = f"""# Compress the output directory
tar -czf ../$(Process)_$(model_config).tar.gz -C {exp_path}/$(Cluster) $(Process)_$(model_config)
ls"""

    hydra_overrides = f'''hydra.run.dir="./{exp_path}/local/hydra" {backslash_char}
hydra.sweep.dir="./{exp_path}/local/hydra" {backslash_char}\n'''
    hydra_overrides_local = f"""'hydra.run.dir=./{exp_path}/local/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}/hydra' {backslash_char}
'hydra.sweep.dir=./{exp_path}/local/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}/hydra'\n
"""
    hydra_overrides_local_launch = [
        f"hydra.run.dir=./{exp_path}/local/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}/hydra",
        f"hydra.sweep.dir=./{exp_path}/local/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}/hydra"
    ]

#     hydra_overrides = f'''hydra.run.dir="./{exp_path}/${{Cluster}}/${{Process}}_${{model_config}}/hydra" {backslash_char}
# hydra.sweep.dir="./{exp_path}/${{Cluster}}/${{Process}}_${{model_config}}/hydra" {backslash_char}\n
# '''

#     hydra_overrides = f'''hydra.run.dir="./{exp_path}/$(Cluster)/$(Process)_$(model_config)/hydra/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}" {backslash_char}
# hydra.sweep.dir="./{exp_path}/$(Cluster)/$(Process)_$(model_config)/hydra/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}" {backslash_char}\n
# '''
    filetree = f"./scripts/create_file_tree.sh ." # dot is for full file tree of the entire repo (except .git)

    # try activating conda
    conda = '''conda init
source /opt/conda/etc/profile.d/conda.sh
conda env list
if conda activate myvenv 2>/dev/null; then
    echo "Using conda env 'myvenv'"
else
    echo "Failed to activate 'myvenv', running rest of script..."
fi\n'''

    # run.sh
    try:
        with open(os.path.join(exp_path, "run.sh"), "w") as file:
            file.write("#!/bin/bash\n")
            file.write(conda)
            file.write(unpack)
            file.write("\n")
            file.write(unpack_data)
            file.write("\n")
            # file.write("tree .")
            # file.write("\n")
            file.write(sub_args_remap)
            file.write("\n")
            file.write(f"cd {project_name}")
            file.write("\n")
            file.write(command)
            file.write(hydra_overrides)
            file.write(chtc_cluster)
            file.write(chtc_process)
            file.write(chtc_model_config)
            file.write("\n")
            file.write('echo "End of python file"')
            file.write("\n")
            file.write('echo')
            file.write("\n")
            file.write("echo $PWD")
            file.write("\n")
            file.write(filetree)
        print(f"Generated {exp_path}/run.sh successfully!")
    except IOError as e:
        print(f"Error writing to {exp_path}: {e}")

    # run.sub
    
    # files created inside exp_path/cluster/process/, including the subdirectories
    output_list = [
        f"hydra/.hydra/config.yaml",
        f"hydra/.hydra/overrides.yaml",
        f"hydra/.hydra/hydra.yaml",
        f"hydra/{os.path.splitext(train_py_root)[1][1:]}.log",
        # f"ckpt/best.pt",
        # f"csv/acc.csv",
        # f"plot/plot1.jpg",
    ]
    sub_outputs_generated = generate_ouput_mappings(output_list, project_name, user_name)


    # Edit for different specs
    vram_presets = [0, 24000, 40000, 45000, 80000] # in mb
    vram = vram_presets[4] # this is hardcoded value
    memory = 100 # 32
    disk = 400 # 100
    cpus = 8 # 2 
    job_length_presets = ['short', 'medium', 'long'] # s=12h, m=24h, l=7d
    job_length = job_length_presets[2] # this is hardcoded value
    
    spec_request = f'''
# Memory, disk and CPU requests
request_memory = {memory}GB
request_disk = {disk}GB
request_cpus = {cpus}

request_gpus = 1
+WantGPULab = true
+GPUJobLength = "{job_length}"
gpus_minimum_capability = 7.0
gpus_minimum_memory = {vram}M
'''
# {"gpus_minimum_memory = " + str(vram) if vram > 0 else ""}
# require_gpus = (Capability >= 7.0){" && (CUDAGlobalMemoryMb >= " + str(vram) + ")" if vram > 0 else ""}


    queue_key, queue_value = create_queue(sub_queue_args)
    sub_id_key = list(sub_id.keys())

    queue_key_map = "# queue_key_map\n"
    queue_key_replace = []
    for i, k in enumerate(queue_key):
        queue_key_replace.append(f"arg{i+1+len(sub_id_key)}")
        queue_key_map += f"arg{i+1+len(sub_id_key)} = {k}\n"

    sub_args_key = sub_id_key + queue_key_replace

    sub_arguments = f'arguments = $({") $(".join(sub_args_key)})\n'
    if len(queue_key)!=0:
        sub_queue=f"queue {', '.join(queue_key_replace)} from (\n"
        for i, comb in enumerate(queue_value):
            row = f"    {', '.join(comb)}\n"
            sub_queue+=row
        sub_queue += "    )"
        
        varkey_value = [f"{x}-$({queue_key_replace[i]})" for i, x in enumerate(queue_key)]
        model_config = f'{"_".join(varkey_value)}'
    else:
        sub_queue="\n"
        # model_config = f'$(Cluster)_$(Process)'
        model_config = f'SingleRun'

    # Replace this with your target directory
    staging_directory = f"/staging/{user_name}/{project_name}"

    # Find all .sif files in the directory
    try:
        sif_files = glob.glob(os.path.join(staging_directory, "apptainers", "*.sif"))
        valid_sif_files = []
        sif_dates=[]
        for f in sif_files:
            sif_base, sif_ext = os.path.splitext(f)
            sif_date = sif_base.rsplit('_', 1)[-1]
            if not sif_date.isnumeric():
                continue
            sif_dates.append(int(sif_date))
            valid_sif_files.append(f)
        sif_latest_idx = sif_dates.index(max(sif_dates))
        sif_latest = valid_sif_files[sif_latest_idx]
        print(f"Using latest .sif {sif_latest}")
    except:
        print(f"Error: No .sif files found in {staging_directory}. Please ensure at least one .sif file is present.")
        sif_latest = "undefined.sif"
    

    pre_query=f'''# Submit file ({exp_path}/run.sub)
model_config={model_config}
number=$(Cluster)_$(Process)
output_directory = {project_name}/{exp_path}/$(Cluster)/$(Process)_$(model_config)
container_local_output_directory = {project_name}/{exp_path}/local
universe = vanilla

log = /home/{user_name}/$(output_directory)/running_detail/job.log
error = /home/{user_name}/$(output_directory)/running_detail/job.err
output = /home/{user_name}/$(output_directory)/running_detail/job.out

container_image = osdf:///chtc{sif_latest}
executable = {project_name}/{exp_path}/run.sh
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

# should add separate directory for the datasets
# transfer_input_files = /staging/{user_name}/{project_name}/{project_tarball} ### since this is already in staging which is shared between container and the server, no need to transfer

'''

    try:
        with open(os.path.join(exp_path, "run.sub"), "w") as file:
            file.write(pre_query)
            file.write("\n")
            file.write(sub_arguments)
            file.write("\n")
            file.write(sub_outputs_generated)
            file.write("\n")
            file.write(spec_request)
            file.write("\n")
            file.write(queue_key_map)
            file.write("\n")
            file.write(sub_queue)
            file.write("\n")
        print(f"Generated {exp_path}/run.sub successfully!")
    except IOError as e:
        print(f"Error writing to {exp_path}: {e}")

    # create run_local.sh
    temp= ''.join(args_local)


    
    local_command = f"python -m {train_py_root} {backslash_char}\n{temp}"

    local_iter_key, local_iter_value = create_queue(args_local_iter)

    # print(f"local command debug : {local_command}")
    # print(f"queue_key: {queue_key}, queue_value: {queue_value}")

    local_commands = []
    launch_configs = []
    for i, iter_value in enumerate(local_iter_value):
        iter_temp = []
        launch_args = launch_args_init.copy()
        # add in ars_local
        for j, iter_key in enumerate(local_iter_key):
            iter_temp.append(f'"{iter_key}={iter_value[j]}" {backslash_char}\n')
            launch_args.append(f"{iter_key}={iter_value[j]}")
        iter_temp= ''.join(iter_temp)
        local_commands.append(f"{local_command}{iter_temp}{hydra_overrides_local}")

        launch_args += hydra_overrides_local_launch
        launch_config = {
            "name": f"{i}",
            "type": "debugpy",
            "request": "launch",
            "module": f"{train_py_root}",
            "console": "integratedTerminal",
            "justMyCode": False,
            "args": launch_args,
        }
        
        launch_configs.append(launch_config)

    try:
        with open(os.path.join(exp_path, "run_local.sh"), "w") as file:
            file.write("#!/bin/bash\n")
            file.write("# This script runs the experiment locally\n")
            file.write("\n")
            for local_command in local_commands:
                file.write(local_command)
                file.write("\n")
        print(f"Generated {exp_path}/run_local.sh successfully!")
    except IOError as e:
        print(f"Error writing to {exp_path}: {e}")


    # launch.json
    os.makedirs(".vscode", exist_ok=True)
    launch_file = os.path.join(".vscode", "launch.json")
    launch_config_template = {
        "version": "0.2.0",
    }
    launch_config_template["configurations"] = launch_configs
    try:
        with open(launch_file, "w", encoding="utf-8") as f:
            json.dump(launch_config_template, f, indent=4)
    except IOError as e:
        print(f"Error writing to {exp_path}: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate run.sh, run.sub from a YAML config file.")
    parser.add_argument("user_name", type=str, help="user name")
    parser.add_argument("project_name", type=str, help="project name")
    parser.add_argument("shared_path", type=str, help="shared path")
    parser.add_argument("experiment_path", type=str, help="Path to the YAML config file")

    args = parser.parse_args()

    config_data = load_config(args.experiment_path)
    if config_data:
        generate_train_script(config_data, args)