#!/usr/bin/env python3
"""
Dataset Management Utility for Keko Pretraining

This script provides utilities to manage pretraining datasets:
- List available datasets
- Inspect dataset contents
- Generate new datasets
- Delete cached datasets
- Compare dataset versions
"""

import argparse
import pickle
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


def list_datasets(datasets_dir: str = "datasets"):
    """List all cached datasets"""
    if not os.path.exists(datasets_dir):
        print(f"⚠ Datasets directory '{datasets_dir}' does not exist")
        return

    pkl_files = list(Path(datasets_dir).glob("*.pkl"))

    if not pkl_files:
        print(f"No datasets found in '{datasets_dir}'")
        return

    print(f"\n{'='*60}")
    print(f"Cached Datasets in '{datasets_dir}'")
    print(f"{'='*60}\n")

    for pkl_file in sorted(pkl_files):
        print(f"📦 {pkl_file.name}")

        # Get file size
        size_mb = pkl_file.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")

        # Get modification time
        mtime = datetime.fromtimestamp(pkl_file.stat().st_mtime)
        print(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

        # Try to load metadata
        metadata_file = pkl_file.with_name(pkl_file.stem + "_metadata.json")
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"   Samples: {metadata.get('sample_count', 'unknown')}")
                print(f"   Created: {metadata.get('created_at', 'unknown')}")
            except Exception as e:
                print(f"   ⚠ Could not read metadata: {e}")

        print()


def inspect_dataset(filepath: str, num_samples: int = 5):
    """Inspect dataset contents"""
    if not os.path.exists(filepath):
        print(f"✗ Dataset not found: {filepath}")
        return

    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}\n")

    try:
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)

        print(f"Total samples: {len(dataset)}")
        print(f"\nShowing first {num_samples} examples:\n")

        for i, (inp, out) in enumerate(dataset[:num_samples]):
            print(f"{'─'*60}")
            print(f"Example {i+1}/{num_samples}")
            print(f"{'─'*60}")
            print(f"INPUT ({len(inp)} chars):")
            print(f"  {inp[:300]}{'...' if len(inp) > 300 else ''}\n")
            print(f"OUTPUT ({len(out)} chars):")
            print(f"  {out[:300]}{'...' if len(out) > 300 else ''}\n")

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")


def delete_dataset(filepath: str, confirm: bool = True):
    """Delete a cached dataset"""
    if not os.path.exists(filepath):
        print(f"⚠ Dataset not found: {filepath}")
        return

    if confirm:
        response = input(f"Delete '{filepath}'? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    try:
        # Delete pickle file
        os.remove(filepath)
        print(f"✓ Deleted: {filepath}")

        # Delete metadata if exists
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            print(f"✓ Deleted: {metadata_file}")

    except Exception as e:
        print(f"✗ Error deleting dataset: {e}")


def analyze_dataset(filepath: str):
    """Analyze dataset statistics"""
    if not os.path.exists(filepath):
        print(f"✗ Dataset not found: {filepath}")
        return

    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {filepath}")
    print(f"{'='*60}\n")

    try:
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)

        # Basic stats
        total_samples = len(dataset)
        input_lengths = [len(inp) for inp, _ in dataset]
        output_lengths = [len(out) for _, out in dataset]

        print(f"📊 Basic Statistics:")
        print(f"   Total samples: {total_samples}")
        print(f"\n📝 Input Statistics:")
        print(f"   Avg length: {sum(input_lengths) / len(input_lengths):.1f} chars")
        print(f"   Min length: {min(input_lengths)} chars")
        print(f"   Max length: {max(input_lengths)} chars")
        print(f"\n📝 Output Statistics:")
        print(f"   Avg length: {sum(output_lengths) / len(output_lengths):.1f} chars")
        print(f"   Min length: {min(output_lengths)} chars")
        print(f"   Max length: {max(output_lengths)} chars")

        # Sample types (heuristic)
        print(f"\n🔍 Content Analysis:")
        multi_turn = sum(1 for inp, _ in dataset if "User:" in inp and "Assistant:" in inp)
        questions = sum(1 for inp, _ in dataset if "?" in inp)
        code_samples = sum(1 for inp, _ in dataset if any(kw in inp.lower() for kw in ['def ', 'class ', 'function', 'import']))

        print(f"   Multi-turn dialogues: {multi_turn} ({multi_turn/total_samples*100:.1f}%)")
        print(f"   Questions: {questions} ({questions/total_samples*100:.1f}%)")
        print(f"   Code-related: {code_samples} ({code_samples/total_samples*100:.1f}%)")

    except Exception as e:
        print(f"✗ Error analyzing dataset: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Manage Keko pretraining datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all datasets
  python manage_datasets.py list

  # Inspect a dataset
  python manage_datasets.py inspect datasets/pretraining_dataset.pkl

  # Analyze dataset statistics
  python manage_datasets.py analyze datasets/pretraining_dataset.pkl

  # Delete a dataset
  python manage_datasets.py delete datasets/old_dataset.pkl

  # Generate new dataset (requires running pretraining.py)
  python pretraining.py --generate-only --dataset-size 5000
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    list_parser = subparsers.add_parser('list', help='List all cached datasets')
    list_parser.add_argument('--dir', default='datasets', help='Datasets directory')

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect dataset contents')
    inspect_parser.add_argument('filepath', help='Path to dataset file')
    inspect_parser.add_argument('--num-samples', type=int, default=5,
                               help='Number of samples to display')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset statistics')
    analyze_parser.add_argument('filepath', help='Path to dataset file')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a cached dataset')
    delete_parser.add_argument('filepath', help='Path to dataset file')
    delete_parser.add_argument('--yes', action='store_true',
                              help='Skip confirmation prompt')

    args = parser.parse_args()

    if args.command == 'list':
        list_datasets(args.dir)
    elif args.command == 'inspect':
        inspect_dataset(args.filepath, args.num_samples)
    elif args.command == 'analyze':
        analyze_dataset(args.filepath)
    elif args.command == 'delete':
        delete_dataset(args.filepath, confirm=not args.yes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
