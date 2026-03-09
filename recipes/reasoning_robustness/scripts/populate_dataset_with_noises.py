#!/usr/bin/env python3
"""
Script to add noise field to all datapoints in MMLU-Redux test.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def add_noise_to_dataset(input_file: str, output_file: str, noise: str):
    """
    Read JSONL file, add noise field to each datapoint, and write to output file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        noise: Noise string to add to each datapoint
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            # Parse JSON
            data = json.loads(line)
            
            # Add noise field
            data['noise'] = noise
            
            # Write modified data
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_count += 1
    
    logging.info(f"Processed {processed_count} datapoints")
    logging.info(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Add noise field to MMLU-Redux test dataset"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='nemo_skills/dataset/mmlu-redux/test.jsonl',
        help='Path to input JSONL file (default: nemo_skills/dataset/mmlu-redux/test.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='nemo_skills/dataset/mmlu-redux/test_with_noise.jsonl',
        help='Path to output JSONL file (default: nemo_skills/dataset/mmlu-redux/test_with_noise.jsonl)'
    )
    parser.add_argument(
        '--noise',
        type=str,
        required=True,
        help='Noise string to add to all datapoints'
    )
    
    args = parser.parse_args()
    
    add_noise_to_dataset(args.input, args.output, args.noise)


if __name__ == '__main__':
    main()
