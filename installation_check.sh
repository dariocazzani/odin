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

# Run tests
echo "Running tests..."
pytest tests/
if [ $? -ne 0 ]; then
    print_error "Tests failed. Initialization aborted."
    exit 1
fi

print_success "Initialization complete. You can now run your Python scripts."
