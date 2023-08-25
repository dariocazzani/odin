#!/bin/bash

source ./scripts/common_functions.sh
display_fancy_header

# Check if the user is in a folder called 'odin'
current_folder=$(basename "$PWD")
if [ "$current_folder" != "odin" ]; then
    echo "Error: You are not in a folder called 'odin'. Please navigate to the correct folder."
    exit 1
fi

# Check if 'run.py' exists in the current folder
if [ ! -f "run.py" ]; then
    echo "Error: 'run.py' does not exist in the current folder. Make sure you are in the correct directory."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH=$(pwd)
echo "PYTHONPATH has been set to: $PYTHONPATH"

# Additional info can be printed here later
# ...

echo "Initialization complete. You can now run your Python scripts."
