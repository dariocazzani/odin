#!/bin/bash

source ./scripts/common_functions.sh
display_fancy_header

# Function to display the manual
display_manual() {
    echo "-------------------------------------------------------------"
    echo "Welcome to the Line Counter!"
    echo "-------------------------------------------------------------"
    echo "Usage:"
    echo "  ./count_lines.sh [OPTIONS] [FILE_EXTENSIONS]"
    echo ""
    echo "Options:"
    echo "  --no-comments     Exclude lines that are comments in Python files."
    echo "  --help or -h      Display this help manual."
    echo ""
    echo "Examples:"
    echo "  ./count_lines.sh .py .rs"
    echo "  ./count_lines.sh --no-comments .py"
    echo "  ./count_lines.sh --help"
    echo ""
    echo "Note: If no file extensions are provided, the script will search for all files."
    echo "-------------------------------------------------------------"
    exit 0
}

# Function to count non-comment lines in a Python file
count_non_comment_lines() {
    local file="$1"
    awk '
    BEGIN { in_comment = 0 }
    /^[\t ]*"""$/ { 
        in_comment = !in_comment; 
        next 
    }
    !in_comment && /^[[:space:]]*#/ { next }
    !in_comment { count++ }
    END { print count }
    ' "$file"
}

exclude_comments=false
file_types=()

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-comments)
            exclude_comments=true
            shift
            ;;
        --help|-h)
            display_manual
            ;;
        *)
            file_types+=("$1")
            shift
            ;;
    esac
done

# Construct the find command
if [ ${#file_types[@]} -eq 0 ]; then
    find_cmd="find . -type f"
else
    find_cmd="find . -type f \( -false"
    for ext in "${file_types[@]}"; do
        find_cmd="$find_cmd -o -name '*$ext'"
    done
    find_cmd="$find_cmd \)"
fi

# Initialize a variable to store the total line count
total_lines=0

# Execute the find command and count lines in each file
while IFS= read -r file; do
    if $exclude_comments; then
        lines=$(count_non_comment_lines "$file")
    else
        lines=$(wc -l "$file" | awk '{print $1}')
    fi
    total_lines=$((total_lines + lines))
done < <(eval $find_cmd)

# Print the total line count
echo "Total lines: $total_lines"
