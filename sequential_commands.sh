#!/bin/bash


source ~/anaconda3/etc/profile.d/conda.sh   # Adjust path if needed

# Activate the desired environment
conda activate uMAIA_env

# List of commands to run
commands=(


  "python uMAIA/peak_finding/extract_images.py   --path_data 'rawData/'   --name '20220620_Zebrafish_atlas_72hpf_fish2_section22_420x142_Att35_7um'"




  
)

# Loop through each command
for cmd in "${commands[@]}"; do
  echo "Running: $cmd"
  eval "$cmd"
  if [ $? -ne 0 ]; then
    echo "Command failed: $cmd"
  else
    echo "Command succeeded: $cmd"
  fi
  echo "-----------------------------"
done