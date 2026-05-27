#!/bin/bash

# place in : /home/user/project/chtc/make_apptainer.sh
# run from : /home/user/project/ bash chtc/make_apptainer.sh [-u]
# creates : /staging/user/project/apptainer.sif


# default values
update=false
# overrides existing requirements.txt with current env

# Currently "run.sh" is set for conda usage.
use_conda=true
use_pip=false


# Process arguments
for arg in "$@"; do
    if [ "$arg" == "-u" ]; then
        update=true
    elif [ "$arg" == "-p" ]; then
        use_conda=false
        use_pip=true
    else
        echo "Warning: Ignoring invalid directory '$arg'"
    fi
done


# update the requirements.txt with current project if -u flag is given
if [ "$update" = true ]; then
    if [ "$use_conda" = true ]; then
        echo "Updating environment.yml"
        # conda env export --no-builds > environment.yml
        conda env export --from-history > environment.yml
        # removing previx
        sed -i '$d' environment.yml
        # Append the pip section
        echo "  - pip" >> environment.yml
        echo "  - pip:" >> environment.yml
        echo "    - -r requirements.txt" >> environment.yml

    fi
    echo "Updating requirements.txt"
    pipreqs --force ./
    echo "hydra-core==1.3.2" >> requirements.txt
else
    echo "Skipping updating requirements.txt"
fi

# get absolute directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get directories
chtc=$(basename "$SCRIPT_DIR")
# project_dir=$(basename "$(dirname "$SCRIPT_DIR")")
# user_dir=$(basename "$(dirname "$(dirname "$SCRIPT_DIR")")")

# Remove leading/trailing slashes
path="${SCRIPT_DIR#/}"
path="${path%/}"

# Split into array
IFS='/' read -ra parts <<< "$path"

len=${#parts[@]}

if (( len < 4 )); then
    echo "Path must have at least 4 hierarchies"
    exit 1
fi

user_dir="${parts[1]}"
project_dir="${parts[len-2]}"

# Extract subpath from 2nd-from-top to 2nd-from-bottom
intermediate_path=$(IFS=/; echo "${parts[*]:1:len-2}")

# debug output
echo "CHTC folder: $chtc"
echo "Project folder: $project_dir"
echo "User folder: $user_dir"
echo "Intermediate path: $intermediate_path"

# make output dir
mkdir -p "/staging/$intermediate_path/apptainers"
echo "Output directory is ready: /staging/$intermediate_path/apptainers"

# make tmp dir
mkdir -p ~/tmp
echo "tmp directory is ready: ~/tmp"

# build apptainer.sif
if [ "$use_conda" = true ]; then
    echo making apptainer with conda
    apptainer build -F --tmpdir ~/tmp \
    /staging/$intermediate_path/apptainers/apptainer_conda_cuda_$(date +%y%m%d%H%M%S).sif \
    $SCRIPT_DIR/apptainer_conda_cuda.def
else
    echo making apptainer with pip
    apptainer build -F --tmpdir ~/tmp \
    /staging/$intermediate_path/apptainers/apptainer_pip_cuda_$(date +%y%m%d%H%M%S).sif \
    $SCRIPT_DIR/apptainer_pip_cuda.def
fi
