#!/bin/bash

# Check if a file list is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <file_list>"
    exit 1
fi

# Read the list of files from the provided argument
file_list=$1

# Function to find the last occurrence of a pattern
find_last_occurrence() {
    tac "$1" | grep -m 1 "$2" | tail -n 1
}

# Loop through each file in the list
while IFS= read -r file; do
    if [ -f "$file" ]; then
        echo "File: $file"

        # Find the last line containing "dirname for harmonic form"
        result=$(find_last_occurrence "$file" "dirname for harmonic form")

        if [ -n "$result" ]; then
            echo "Last 'dirname for harmonic form' line: $result"
        else
            # If not found, look for "dirname for beta"
            result=$(find_last_occurrence "$file" "dirname for beta")
            if [ -n "$result" ]; then
                echo "Last 'dirname for beta' line: $result"
            else
                echo "Neither 'dirname for harmonic form' nor 'dirname for beta' found"
            fi
        fi

        # Check if "Compute holomorphic Yukawas" is in the last 100 lines
        if tail -n 100 "$file" | grep -q "Compute holomorphic Yukawas"; then
            echo "Status: Finished more-or-less successfully"
        else
            echo "Status: Did not finish successfully"
        fi

        echo "------------------------"
    else
        echo "File not found: $file"
        echo "------------------------"
    fi
done < "$file_list"
