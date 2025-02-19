#!/bin/bash

find . -type f | while read -r file; do
    if head -n 10 "$file" | grep -q "gpu09"; then
        size_kb=$(du -k "$file" | cut -f1)
        echo "$file - ${size_kb} KB"
    fi
done

