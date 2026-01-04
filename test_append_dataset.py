"""Test script for dataset append functionality."""
import os
import tempfile
import shutil
from pathlib import Path
from dataset_generator.generation import GenerationConfig, generate_factset

def test_append_mode():
    """Test that append mode correctly combines datasets."""

    # Create a temporary directory for test output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_dataset"

        # Part 1: Create initial dataset with 5 prompts
        print("\n" + "="*60)
        print("PART 1: Creating initial dataset with 5 prompts")
        print("="*60)

        prompts_part1 = [
            "What is the exact color of happiness?",
            "What is the exact weight of a dream?",
            "What is the exact taste of freedom?",
        ]

        cfg1 = GenerationConfig(
            checkpoint_path="pretrained_columns.pt",
            prompts=prompts_part1,
            max_new_tokens=512,
            num_samples=None,  # Use all prompts
            device="cuda",
            output_path=str(output_path),
            save_json=True,
            save_jsonl=True,
            num_workers=1,
            append_mode=False  # Create new dataset
        )

        generate_factset(cfg1)

        # Check initial dataset size
        from datasets import Dataset
        dataset1 = Dataset.load_from_disk(str(output_path))
        print(f"\n✅ Initial dataset created: {len(dataset1)} records")

        # Part 2: Append more prompts
        print("\n" + "="*60)
        print("PART 2: Appending 3 more prompts to existing dataset")
        print("="*60)

        prompts_part2 = [
            "What is the exact sound of silence?",
            "What is the exact shape of time?",
        ]

        cfg2 = GenerationConfig(
            checkpoint_path="pretrained_columns.pt",
            prompts=prompts_part2,
            max_new_tokens=512,
            num_samples=None,
            device="cuda",
            output_path=str(output_path),
            save_json=True,
            save_jsonl=True,
            num_workers=1,
            append_mode=True  # Append to existing
        )

        generate_factset(cfg2)

        # Check final dataset size
        dataset2 = Dataset.load_from_disk(str(output_path))
        print(f"\n✅ Final dataset: {len(dataset2)} records")

        # Verify the append worked correctly
        expected_total = len(prompts_part1) + len(prompts_part2)
        assert len(dataset2) == expected_total, \
            f"Expected {expected_total} records, got {len(dataset2)}"

        # Verify prompts from both parts are present
        all_prompts = [record['prompt'] for record in dataset2]
        for prompt in prompts_part1 + prompts_part2:
            assert prompt in all_prompts, f"Missing prompt: {prompt}"

        print("\n" + "="*60)
        print("✅ SUCCESS: Append mode test passed!")
        print(f"   - Part 1: {len(prompts_part1)} prompts")
        print(f"   - Part 2: {len(prompts_part2)} prompts")
        print(f"   - Total: {len(dataset2)} records")
        print("="*60)

        # Show sample records
        print("\n📋 Sample records from final dataset:")
        for i, record in enumerate(dataset2):
            print(f"\n{i+1}. Prompt: {record['prompt']}")
            print(f"   Response: {record['response']}")
            print(f"   Confidence: {record['mean_confidence']:.3f}")

if __name__ == "__main__":
    test_append_mode()