#!/bin/bash

# Function to print errors in red
print_error() {
    printf "\e[31mError: $1\e[0m\n"
}

# Function to print warnings in yellow
print_warning() {
    printf "\e[33mWarning: $1\e[0m\n"
}

# Function to print success messages in green
print_success() {
    printf "\e[32m$1\e[0m\n"
}

# Source the common functions
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$script_dir/scripts/common_functions.sh"

# Display the fancy header
display_fancy_header

# Check if the user is using a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    print_error "You are not using a Python virtual environment. Please activate one before proceeding."
    exit 1
fi

# Check if the user is in a folder called 'odin'
current_folder=$(basename "$PWD")
if [ "$current_folder" != "odin" ]; then
    print_error "You are not in a folder called 'odin'. Please navigate to the correct folder."
    exit 1
fi

# Check if 'run.py' exists in the current folder
if [ ! -f "run.py" ]; then
    print_error "'run.py' does not exist in the current folder. Make sure you are in the correct directory."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH=$(pwd)
echo "PYTHONPATH has been set to: $PYTHONPATH"

# Check if pre-commit is installed, and install it if not
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit package..."
    pip install pre-commit
else
    print_success "pre-commit is already installed."
fi

# Check if pre-commit hooks are installed, and install them if not
if [ ! -f ".git/hooks/pre-commit" ]; then
    print_warning "Installing pre-commit hooks..."
    pre-commit install -f
else
    print_success "pre-commit hooks are already installed."
fi


# Run tests
echo "Running tests..."
pytest tests/
if [ $? -ne 0 ]; then
    print_error "Tests failed. Initialization aborted."
    exit 1
fi

# Additional info can be printed here later
# ...

print_success "Initialization complete. You can now run your Python scripts."
