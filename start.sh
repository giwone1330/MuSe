#!/bin/bash

# Check if at least one argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <runs_subfolder> [-c] [-s] [-l] [-v]"
    exit 1
fi

# default values for options
compress=false
compress_data=false
skip=false
local=false # not sure if running should be done by command
no_run=false
verbose=false
DIRS=()

# Helper function to add a directory if not already in DIRS
add_dir() {
    local new_dir="$1"
    for existing in "${DIRS[@]}"; do
        if [[ "$existing" == "$new_dir" ]]; then
            echo "'$new_dir' already in list, skipping duplicate"
            return 0  # already exists, skip
        fi
    done
    DIRS+=("$new_dir")
}

# Process arguments
for arg in "$@"; do
    if [ "$arg" == "-c" ]; then
        compress=true
    elif [ "$arg" == "-s" ]; then
        skip=true
    elif [ "$arg" == "-d" ]; then
        compress_data=true
    elif [ "$arg" == "-l" ]; then
        local=true
    elif [ "$arg" == "-n" ]; then
        no_run=true
    elif [ "$arg" == "-v" ]; then
        verbose=true
    elif [ -d "$arg" ]; then
        # Also find all subdirectories under $arg containing run.yaml
        while IFS= read -r d; do
            if [ -f "$d/run.yaml" ]; then
                add_dir "$d"
            else
                echo "Warning: Skipping '$arg' because 'run.yaml' was not found directly inside."
            fi
        done < <(
            find "$arg" -type f -name "run.yaml" \
            | xargs -r -n1 dirname \
            | sort -u
        )
    else
        echo "Warning: Ignoring invalid directory '$arg'"
    fi
done

# Ensure at least one valid directory was provided
if [ ${#DIRS[@]} -eq 0 ]; then
    echo "Error: No valid directories containing 'run.yaml' were provided."
    exit 1
fi


# get absolute directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remove leading/trailing slashes
path="${SCRIPT_DIR#/}"
path="${path%/}"

# Split into array
IFS='/' read -ra parts <<< "$path"
len=${#parts[@]}
if (( len < 3 )); then
    echo "Path must have at least 3 hierarchies"
    exit 1
fi

user_dir="${parts[1]}"
project_dir="${parts[len-1]}"

# Extract subpath from 2nd-from-top to 1st-from-bottom
shared_path=$(IFS=/; echo "${parts[*]:1:len-1}")

# # get absolute directory of the script
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# # get directories
# project_dir=$(basename "$SCRIPT_DIR")
# user_dir=$(basename "$(dirname "$SCRIPT_DIR")")

# check for exsitance of teaching_arithmetic.tar.gz
if [ -f "/staging/$shared_path/$project_dir.tar.gz" ]; then
    echo "✅ /staging/$shared_path/$project_dir.tar.gz exists"
else
    echo "❌ /staging/$shared_path/$project_dir.tar.gz missing"
    compress=true
fi

# Display options
echo "Directories: ${DIRS[*]}"
echo "Compression: $compress"
echo "Skip script gen: $skip"
echo "local: $local"
# debug output
echo "Project folder: $project_dir"
echo "User folder: $user_dir"
echo "Shared path: $shared_path"


# compress/overwrite .tar.gz
if $compress; then
    archive_dir="/staging/$shared_path"
    archive_prefix="${project_dir}_"

    # Delete existing tarballs that start with $project_dir_
    find "$archive_dir" -maxdepth 1 -type f -name "${archive_prefix}*.tar.gz" -delete

    # Create new archive
    tar -czf "$archive_dir/${project_dir}_$(date +%y%m%d%H%M%S).tar.gz" \
        -C "/home/$user_dir" \
        --exclude="$project_dir/data" \
        "$project_dir"

    echo "compression done"
fi

if $compress_data; then
    archive_dir="/staging/$shared_path"
    archive_prefix="data_"

    # Delete existing data_*.tar.gz archives
    find "$archive_dir" -maxdepth 1 -type f -name "${archive_prefix}*.tar.gz" -delete

    # Create new archive
    tar -czf "$archive_dir/data_$(date +%y%m%d%H%M%S).tar.gz" \
        -C "/home/$user_dir" \
        "$project_dir/data"

    echo "data compression done"
fi


# Run Python create_exp_script command for each directory
for dir in "${DIRS[@]}"; do

    if $skip; then
        # Check for run.sh and run.sub
        if [ -f "$dir/run.sh" ] && [ -f "$dir/run.sub" ]; then
            echo "✅ Both run.sh and run.sub exist in '$dir'."
            echo "Safely skipping the script creation"
        else
            echo "❌ One or both files are missing in '$dir'."
            echo "Creating/overwriting run.sh, run.sub"
            python scripts/create_exp_script.py "$user_dir" "$project_dir" "$shared_path" "$dir"
        fi
    else
        # create .sh, .sub if skip == false
        echo "Creating/overwriting run.sh, run.sub"
        # python create_exp_script.py --config-path "$dir" hydra.output_subdir=null hydra.job_logging=none hydra.hydra_logging=none
        python scripts/create_exp_script.py "$user_dir" "$project_dir" "$shared_path" "$dir"
    fi
done

if $no_run; then
    echo "no_run=true, exiting before starting any runs." 
    exit 0
fi



# if local is true, exit here
if $local; then
    echo "Local flag is set!, skipping chtc submission"
    for dir in "${DIRS[@]}"; do
        bash "$dir/run_local.sh"
        echo "experiment '$dir' started locally"
    done
    exit 0
fi

# move to home directory and submit jobs (easier to manage output file transfer)
cd ..
# Run submit command for each directory
for dir in "${DIRS[@]}"; do
    condor_submit "$project_dir/$dir/run.sub"
    echo "experiment '$dir' submitted"
done
