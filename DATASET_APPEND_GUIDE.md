# Dataset Append Mode Guide

This guide explains how to build datasets incrementally by appending prompts in parts.

## Overview

The dataset generator now supports **append mode**, allowing you to:
- Build datasets incrementally without regenerating everything
- Add prompts in batches as you create them
- Combine multiple prompt files into a single dataset
- Resume dataset generation after interruptions

## Quick Start

### Example 1: Building a Dataset in Parts

```bash
# Step 1: Create initial dataset with first batch of prompts
python -m dataset_generator.cli \
  --output-path datasets/my_dataset \
  --prompts-file prompts_batch1.txt \
  --num-samples 50

# Step 2: Append second batch (notice --append flag)
python -m dataset_generator.cli \
  --output-path datasets/my_dataset \
  --prompts-file prompts_batch2.txt \
  --num-samples 50 \
  --append

# Step 3: Append third batch
python -m dataset_generator.cli \
  --output-path datasets/my_dataset \
  --prompts-file prompts_batch3.txt \
  --num-samples 50 \
  --append
```

### Example 2: Interactive Dataset Building

Create a prompt file with just a few prompts at a time:

```bash
# Create prompts_1.txt with 10 prompts
cat > prompts_1.txt << EOF
What is the exact color of love?
What is the exact weight of time?
EOF

# Generate initial dataset
python -m dataset_generator.cli \
  --prompts-file prompts_1.txt \
  --output-path datasets/interactive_dataset

# Later, add more prompts
cat > prompts_2.txt << EOF
What is the exact taste of happiness?
What is the exact shape of a dream?
EOF

# Append to existing dataset
python -m dataset_generator.cli \
  --prompts-file prompts_2.txt \
  --output-path datasets/interactive_dataset \
  --append
```

### Example 3: Using Default Prompts

```bash
# Create initial dataset from first 100 impossible prompts
python -m dataset_generator.cli \
  --output-path datasets/impossible_questions \
  --num-samples 100

# Later, append next 100 prompts (101-200)
# Note: You'll need to manage which prompts to use
python -m dataset_generator.cli \
  --output-path datasets/impossible_questions \
  --num-samples 100 \
  --append
```

## Command-Line Arguments

```bash
python -m dataset_generator.cli [options]

Required:
  --output-path PATH        Directory to save the dataset

Optional:
  --prompts-file FILE       Newline-delimited text file with prompts
                           (if not provided, uses IMPOSSIBLE_PROMPTS)
  --num-samples N           Limit to first N prompts (default: 10)
  --append                  Append to existing dataset (default: overwrite)
  --max-tokens N            Max tokens per response (default: 2048)
  --temperature T           Sampling temperature (default: 0.7)
  --top-k K                 Top-k sampling (default: 30)
  --device DEVICE           cuda or cpu (default: cuda)
  --num-workers N           Parallel workers (default: 2)
```

## Prompt File Format

Create a simple text file with one prompt per line:

```text
What is the exact color of happiness?
What is the exact weight of a dream?
What is the exact taste of freedom?
What is the exact sound of silence?
```

Blank lines are automatically skipped.

## How Append Mode Works

1. **First run (no --append)**:
   - Creates a new dataset
   - Overwrites any existing dataset at the output path

2. **Subsequent runs (with --append)**:
   - Loads existing dataset from the output path
   - Generates responses for new prompts
   - Combines old and new records
   - Saves the combined dataset

## Checking Dataset Size

```bash
# View dataset info
python -c "
from datasets import Dataset
ds = Dataset.load_from_disk('datasets/my_dataset')
print(f'Total records: {len(ds)}')
print(f'Sample record: {ds[0]}')
"

# Count records in JSONL file
wc -l datasets/my_dataset/dataset.jsonl
```

## Programmatic Usage

```python
from dataset_generator import GenerationConfig, generate_factset

# Part 1: Initial dataset
prompts_part1 = ["What is X?", "What is Y?"]
cfg1 = GenerationConfig(
    checkpoint_path="pretrained_columns.pt",
    prompts=prompts_part1,
    output_path="datasets/my_dataset",
    append_mode=False  # Create new
)
generate_factset(cfg1)

# Part 2: Append more prompts
prompts_part2 = ["What is Z?", "What is W?"]
cfg2 = GenerationConfig(
    checkpoint_path="pretrained_columns.pt",
    prompts=prompts_part2,
    output_path="datasets/my_dataset",
    append_mode=True  # Append to existing
)
generate_factset(cfg2)

# Load final dataset
from datasets import Dataset
final_dataset = Dataset.load_from_disk("datasets/my_dataset")
print(f"Total records: {len(final_dataset)}")
```

## Best Practices

### 1. Start Small, Grow Incrementally
```bash
# Start with 10 prompts to test
python -m dataset_generator.cli \
  --prompts-file test_prompts.txt \
  --num-samples 10 \
  --output-path datasets/test

# Once satisfied, add more
python -m dataset_generator.cli \
  --prompts-file more_prompts.txt \
  --output-path datasets/test \
  --append
```

### 2. Keep Prompt Files Organized
```
prompts/
├── batch_01_philosophy.txt     (50 prompts)
├── batch_02_science.txt        (50 prompts)
├── batch_03_emotions.txt       (50 prompts)
└── batch_04_abstract.txt       (50 prompts)
```

Generate incrementally:
```bash
for batch in prompts/batch_*.txt; do
  python -m dataset_generator.cli \
    --prompts-file "$batch" \
    --output-path datasets/comprehensive \
    --append
done
```

### 3. Backup Before Appending
```bash
# Backup existing dataset
cp -r datasets/my_dataset datasets/my_dataset.backup

# Then append
python -m dataset_generator.cli \
  --output-path datasets/my_dataset \
  --prompts-file new_prompts.txt \
  --append
```

### 4. Track What You've Added
Keep a log file:
```bash
echo "$(date): Added 50 prompts from batch_01.txt" >> datasets/my_dataset/generation_log.txt
```

## Troubleshooting

### "Warning: Could not load existing dataset for append"

This happens if:
- The output path doesn't exist yet (will create new dataset)
- The dataset files are corrupted
- The dataset format is incompatible

**Solution**: Remove `--append` flag to create a new dataset.

### Duplicate Prompts

The system doesn't automatically deduplicate prompts. To avoid duplicates:

```python
from datasets import Dataset

# Load existing dataset
ds = Dataset.load_from_disk("datasets/my_dataset")
existing_prompts = set(record['prompt'] for record in ds)

# Filter new prompts
new_prompts = [p for p in your_prompts if p not in existing_prompts]
```

### Dataset Growing Too Large

Check size:
```bash
du -sh datasets/my_dataset/
```

If too large, consider:
- Splitting into multiple datasets by topic
- Filtering low-quality responses
- Reducing `--max-tokens`

## Testing

Run the test script to verify append functionality:

```bash
python test_append_dataset.py
```

This will:
1. Create a dataset with 3 prompts
2. Append 2 more prompts
3. Verify total is 5 records
4. Display sample records

## Related Documentation

- [DATASET_MANAGEMENT.md](DATASET_MANAGEMENT.md) - Main dataset management guide
- [dataset_generator/generation.py](dataset_generator/generation.py) - Core generation code
- [dataset_generator/cli.py](dataset_generator/cli.py) - CLI implementation

## Examples in Practice

### Use Case: Daily Dataset Growth

```bash
# Monday: Start with philosophical questions
python -m dataset_generator.cli \
  --prompts-file monday_philosophy.txt \
  --output-path datasets/weekly_build

# Tuesday: Add scientific questions
python -m dataset_generator.cli \
  --prompts-file tuesday_science.txt \
  --output-path datasets/weekly_build \
  --append

# Wednesday: Add emotional questions
python -m dataset_generator.cli \
  --prompts-file wednesday_emotions.txt \
  --output-path datasets/weekly_build \
  --append

# ... continue daily
```

### Use Case: Collaborative Dataset Building

Team members can contribute prompts:

```bash
# Team member 1
python -m dataset_generator.cli \
  --prompts-file alice_prompts.txt \
  --output-path datasets/team_dataset

# Team member 2 (appends)
python -m dataset_generator.cli \
  --prompts-file bob_prompts.txt \
  --output-path datasets/team_dataset \
  --append

# Team member 3 (appends)
python -m dataset_generator.cli \
  --prompts-file charlie_prompts.txt \
  --output-path datasets/team_dataset \
  --append
```

## Summary

The `--append` flag enables flexible, incremental dataset building:

- ✅ **Create once**: `python -m dataset_generator.cli --output-path datasets/my_dataset`
- ✅ **Append many times**: Add `--append` flag for subsequent runs
- ✅ **Use custom prompts**: `--prompts-file your_prompts.txt`
- ✅ **Track progress**: Dataset grows with each append operation

Happy dataset building! 🎯
