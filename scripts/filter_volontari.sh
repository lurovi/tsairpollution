#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.csv output.csv"
    exit 1
fi

input_file="$1"
output_file="$2"

# Extract header + matching rows
(head -n 1 "$input_file" && grep 'Trieste - P.zza Volontari Giuliani' "$input_file") > "$output_file"

echo "Filtered rows (with header) saved to $output_file"

