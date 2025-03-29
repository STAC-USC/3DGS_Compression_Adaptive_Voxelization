#!/bin/bash

# Define the parent directory and create directories for ply files
PARENT_DIR="C:/Users/jay/Desktop/Ablation"

# Function to handle interruption and output debug information
function handle_interrupt {
    echo "Script interrupted. Debugging information:"
    echo "Current working directory: $(pwd)"
    echo "Files in current directory:"
    ls -l
    exit 1
}

# Trap interrupt signals
trap handle_interrupt SIGINT SIGTERM

# Traverse the parent directory and process ply files
for SUBDIR in "$PARENT_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        echo "Processing subdirectory: $SUBDIR"
        for PLY_FILE in "$SUBDIR"/*.ply; do
            if [ -f "$PLY_FILE" ]; then
                FILE_NAME=$(basename "$PLY_FILE" .ply)
                FILE_DIR="$SUBDIR/$FILE_NAME"
                echo "Creating directory: $FILE_DIR"
                mkdir -p "$FILE_DIR"
                echo "Moving $PLY_FILE to $FILE_DIR"
                mv "$PLY_FILE" "$FILE_DIR/"
            fi
        done
    fi
done

echo "All files processed."
