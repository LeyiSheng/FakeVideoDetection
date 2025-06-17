#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modify the path prefix in the label.csv file
"""

import os
import argparse

def update_label_paths(input_file, output_file, old_prefix, new_prefix):
    """
    Update the path prefix in the label.csv file
    
    Args:
        input_file: The path of the input label.csv file
        output_file: The path of the output label.csv file
        old_prefix: The old path prefix to be replaced
        new_prefix: The new path prefix
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    # Remove the newline character at the end of the line
                    line = line.strip()
                    if line:
                        # Replace the path prefix
                        if line.startswith(old_prefix):
                            updated_line = line.replace(old_prefix, new_prefix, 1)
                            outfile.write(updated_line + '\n')
                        else:
                            # If the old prefix is not found, keep the original line
                            outfile.write(line + '\n')
        
        print(f"Path update completed!")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Old prefix: {old_prefix}")
        print(f"New prefix: {new_prefix}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found {input_file}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    # python update_label_paths.py -i FakeAVCeleb/label.csv --old-prefix "/your_path/" --new-prefix "/your/new/path/"
    parser = argparse.ArgumentParser(description='Modify the path prefix in the label.csv file')
    parser.add_argument('--input', '-i', required=True, help='The path of the input label.csv file')
    parser.add_argument('--output', '-o', help='The path of the output label.csv file (default: input file name_updated.csv)')
    parser.add_argument('--old-prefix', required=True, help='The old path prefix to be replaced')
    parser.add_argument('--new-prefix', required=True, help='The new path prefix')
    
    args = parser.parse_args()
    
    # If no output file is specified, default to adding _updated to the input file name
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_updated.csv"
    
    update_label_paths(args.input, args.output, args.old_prefix, args.new_prefix)

if __name__ == "__main__":
    main() 