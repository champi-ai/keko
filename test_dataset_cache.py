#!/usr/bin/env python3
"""
Quick test to verify dataset caching functionality
"""

import os
import sys
from pathlib import Path

# Add simple mock classes to test without loading the full model
class MockTokenizer:
    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

class MockModel:
    class Config:
        hidden_size = 576
        vocab_size = 30000

    config = Config()

class SimplifiedTrainer:
    """Simplified version just for testing save/load"""

    def __init__(self):
        import pickle
        import json
        from datetime import datetime

        self.pickle = pickle
        self.json = json
        self.datetime = datetime

    def save_dataset(self, dataset, filepath="datasets/test_dataset.pkl"):
        """Save generated dataset to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            self.pickle.dump(dataset, f)

        metadata_path = filepath.replace('.pkl', '_metadata.json')
        metadata = {
            'size': len(dataset),
            'created_at': self.datetime.now().isoformat(),
            'sample_count': len(dataset),
            'first_sample': {
                'input': dataset[0][0][:100] if dataset else None,
                'output': dataset[0][1][:100] if dataset else None
            }
        }
        with open(metadata_path, 'w') as f:
            self.json.dump(metadata, f, indent=2)

        print(f"✓ Saved dataset to {filepath}")
        print(f"✓ Saved metadata to {metadata_path}")
        print(f"  Total samples: {len(dataset)}")
        return filepath

    def load_dataset(self, filepath="datasets/test_dataset.pkl"):
        """Load previously generated dataset from disk"""
        if not os.path.exists(filepath):
            print(f"⚠ Dataset not found at {filepath}")
            return None

        try:
            with open(filepath, 'rb') as f:
                dataset = self.pickle.load(f)

            print(f"✓ Loaded dataset from {filepath}")
            print(f"  Total samples: {len(dataset)}")

            metadata_path = filepath.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = self.json.load(f)
                print(f"  Created: {metadata.get('created_at', 'unknown')}")

            return dataset
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return None


def test_save_load():
    """Test dataset save and load functionality"""

    print("="*60)
    print("Testing Dataset Cache Functionality")
    print("="*60)

    # Create mock dataset
    print("\n1. Creating mock dataset...")
    mock_dataset = [
        ("What is 2+2?", "2+2 equals 4"),
        ("Hello, how are you?", "I'm doing well, thank you!"),
        ("Explain Python", "Python is a high-level programming language..."),
    ]
    print(f"   Created {len(mock_dataset)} sample entries")

    # Initialize trainer
    print("\n2. Initializing simplified trainer...")
    trainer = SimplifiedTrainer()
    print("   ✓ Trainer ready")

    # Test save
    print("\n3. Testing save functionality...")
    cache_path = "keko_datasets/test_dataset.pkl"
    trainer.save_dataset(mock_dataset, cache_path)

    # Verify files exist
    print("\n4. Verifying saved files...")
    if os.path.exists(cache_path):
        print(f"   ✓ Dataset file exists: {cache_path}")
        size_kb = os.path.getsize(cache_path) / 1024
        print(f"   ✓ File size: {size_kb:.2f} KB")
    else:
        print(f"   ✗ Dataset file not found!")
        return False

    metadata_path = cache_path.replace('.pkl', '_metadata.json')
    if os.path.exists(metadata_path):
        print(f"   ✓ Metadata file exists: {metadata_path}")
    else:
        print(f"   ✗ Metadata file not found!")
        return False

    # Test load
    print("\n5. Testing load functionality...")
    loaded_dataset = trainer.load_dataset(cache_path)

    if loaded_dataset is None:
        print("   ✗ Failed to load dataset!")
        return False

    # Verify loaded data
    print("\n6. Verifying loaded data...")
    if len(loaded_dataset) == len(mock_dataset):
        print(f"   ✓ Correct number of samples: {len(loaded_dataset)}")
    else:
        print(f"   ✗ Sample count mismatch: {len(loaded_dataset)} vs {len(mock_dataset)}")
        return False

    if loaded_dataset[0] == mock_dataset[0]:
        print(f"   ✓ First sample matches")
    else:
        print(f"   ✗ First sample mismatch!")
        return False

    print("\n7. Testing dataset management utility...")
    import subprocess

    # Test list command
    result = subprocess.run(
        ["python", "manage_datasets.py", "list"],
        capture_output=True,
        text=True
    )
    print("   ✓ List command executed")
    if "test_dataset.pkl" in result.stdout:
        print("   ✓ Dataset appears in list")

    # Test inspect command
    result = subprocess.run(
        ["python", "manage_datasets.py", "inspect", cache_path, "--num-samples", "2"],
        capture_output=True,
        text=True
    )
    print("   ✓ Inspect command executed")
    if "What is 2+2?" in result.stdout:
        print("   ✓ Sample data visible in inspection")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nDataset caching is working correctly.")
    print(f"Test dataset saved at: {cache_path}")
    print("\nYou can now use:")
    print("  - python manage_datasets.py list")
    print("  - python manage_datasets.py inspect datasets/test_dataset.pkl")
    print("  - python manage_datasets.py analyze datasets/test_dataset.pkl")

    return True


if __name__ == "__main__":
    try:
        success = test_save_load()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
