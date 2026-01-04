# Testing Guide for Keko PNN

This guide covers how to test pretrained columns and validate training results.

## Quick Start

### 1. Analyze a Checkpoint
```bash
python test_inference.py --mode analyze
```

**Output:**
- Checkpoint contents and structure
- Model dimensions (hidden size, vocab size)
- Training information
- Parameter counts
- File size

### 2. Compare Base vs Columns
```bash
python test_inference.py --mode compare --prompt "What is AI?"
```

**Shows:**
- Base model's top-5 token predictions
- Each column's top-5 token predictions
- Side-by-side comparison

### 3. Full Inference Test
```bash
python test_inference.py --mode test --prompt "Hello, how are you?"
```

**Tests:**
- Base model generation (full text)
- Column-enhanced next token prediction
- Multiple test prompts

## Test Modes

### Mode: `analyze`
**Purpose:** Inspect checkpoint without loading the full model

**Usage:**
```bash
python test_inference.py --mode analyze --checkpoint pretrained_columns.pt
```

**When to use:**
- Quick checkpoint validation
- Check training time/parameters
- Verify file integrity

**Example output:**
```
📊 Checkpoint Contents:
   Keys: ['columns_state_dict', 'output_projection_state_dict', ...]

📐 Model Dimensions:
   Hidden size: 960
   Vocabulary size: 49152

⏱️  Training Info:
   Training time: 3.38 minutes
   Device: cuda

🏗️  Column Architecture:
   Parameters per column: 2.77M
```

### Mode: `compare`
**Purpose:** Compare base model vs column predictions

**Usage:**
```bash
python test_inference.py --mode compare --prompt "Your question here"
```

**When to use:**
- Validate column training
- Check if columns learned something different from base
- Debug NaN or unusual outputs

**Healthy output example:**
```
🔹 Base Model Logits:
   Top 5 tokens:
      1.5557 → '\n'
      1.4199 → '<|im_end|>'
      0.5200 → ' ('

🔸 Column 0 Logits:
   Top 5 tokens:
      1.6231 → '\n'
      1.3452 → '<|im_end|>'
      0.6134 → ' ('
```

**Unhealthy output (needs more training):**
```
🔸 Column 0 Logits:
   Top 5 tokens:
      nan → '<|im_start|>'
      nan → '<|endoftext|>'
```

### Mode: `test`
**Purpose:** Full inference pipeline test

**Usage:**
```bash
# Single prompt
python test_inference.py --mode test --prompt "What is Python?"

# Default prompts (5 predefined)
python test_inference.py --mode test

# Control generation length
python test_inference.py --mode test --max-length 100
```

**When to use:**
- End-to-end validation
- Compare full generations
- Test multiple prompts

## Understanding Test Results

### ✅ Good Results

**Base Model:**
```
Prompt: What is 2 + 2?
🔹 Base Model Only:
   2 + 2 equals 4. This is basic arithmetic...
```
✓ Coherent, relevant response

**Columns:**
```
🔸 Column 0 Logits:
   Top 5 tokens:
      1.2345 → ' equals'
      1.1234 → ' is'
      0.9876 → ' ='
```
✓ Real numbers (not NaN)
✓ Relevant tokens for the prompt

### ❌ Bad Results (Need More Training)

**Columns producing NaN:**
```
🔸 Column 0 Logits:
   Top 5 tokens:
      nan → '<|im_start|>'
      nan → '<|endoftext|>'
```
**Cause:** Insufficient training (e.g., 1 epoch on 10 samples)
**Fix:** Train longer with more data

**Columns producing only EOS tokens:**
```
Column 0 next token: '<|endoftext|>'
Column 1 next token: '<|endoftext|>'
Column 2 next token: '<|endoftext|>'
```
**Cause:** Columns learned to end generation immediately
**Fix:** Adjust training loss or increase diversity

## Training Requirements for Valid Testing

### Minimum for Testing
```bash
python pretraining.py --epochs 3 --batch-size 4 --dataset-size 1000
```
**Time:** ~10-15 minutes
**Result:** Columns should produce valid (non-NaN) outputs

### Recommended for Good Results
```bash
python pretraining.py --epochs 5 --batch-size 8 --dataset-size 5000
```
**Time:** ~30-45 minutes
**Result:** Columns should show some specialization

### Full Training
```bash
python pretraining.py --epochs 10 --batch-size 8 --dataset-size 10000
```
**Time:** ~2-3 hours
**Result:** Columns should be well-initialized for specialization

## Workflow Example

### 1. Quick Test (Fast)
```bash
# Train minimally
python pretraining.py --epochs 1 --batch-size 2 --dataset-size 100

# Analyze checkpoint
python test_inference.py --mode analyze

# Quick comparison
python test_inference.py --mode compare --prompt "Hello"
```
**Expected:** Likely to see NaN values, but validates pipeline

### 2. Validation Test (Medium)
```bash
# Train with moderate settings
python pretraining.py --epochs 3 --batch-size 4 --dataset-size 1000

# Compare outputs
python test_inference.py --mode compare --prompt "What is AI?"

# Full test
python test_inference.py --mode test
```
**Expected:** Valid outputs, may still be similar to base model

### 3. Production Test (Thorough)
```bash
# Full training
python pretraining.py --epochs 10 --batch-size 8 --dataset-size 10000

# Test with multiple prompts
python test_inference.py --mode test

# Compare specific prompts
python test_inference.py --mode compare --prompt "Explain quantum computing"
```
**Expected:** Columns show differentiation from base model

## Troubleshooting

### Issue: NaN values in column outputs

**Symptoms:**
```
🔸 Column 0 Logits:
   Top 5 tokens:
      nan → '<|im_start|>'
```

**Diagnosis:**
```bash
python test_inference.py --mode analyze
# Check: Training time < 5 minutes? Too short!
```

**Solutions:**
1. **Train longer:**
   ```bash
   python pretraining.py --epochs 5 --dataset-size 5000
   ```

2. **Check loss during training:**
   - Loss should decrease
   - If loss is NaN during training → learning rate too high

3. **Verify dtype:**
   - All components should be float16
   - Check with `--mode analyze`

### Issue: CUDA out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Reduce batch size:**
   ```bash
   python pretraining.py --batch-size 2
   ```

2. **Reduce dataset size during inference:**
   ```bash
   # Test with single prompt
   python test_inference.py --prompt "Short test"
   ```

3. **Use CPU (slow):**
   ```bash
   # Modify test_inference.py: device = 'cpu'
   ```

### Issue: Columns output same as base model

**Symptoms:**
```
🔹 Base: 'Hello'
🔸 Column 0: 'Hello'
🔸 Column 1: 'Hello'
```

**Diagnosis:**
This is actually **expected** for pretrained columns! They're trained to match the base model's distribution (KL divergence loss).

**Explanation:**
- Pretraining creates "fertile ground"
- Columns should match base initially
- Specialization happens during uncertainty-driven training (uncertain.py)

**Not a bug if:**
- Using checkpoints from `pretraining.py`
- Haven't run `uncertain.py` yet
- This is the "fertile ground" phase

### Issue: "Checkpoint not found"

**Symptoms:**
```
FileNotFoundError: Checkpoint not found: pretrained_columns.pt
```

**Solutions:**
1. **Train first:**
   ```bash
   python pretraining.py --epochs 1 --dataset-size 100
   ```

2. **Specify checkpoint path:**
   ```bash
   python test_inference.py --checkpoint path/to/checkpoint.pt
   ```

3. **Check if training completed:**
   ```bash
   ls -lh *.pt
   # Should see pretrained_columns.pt
   ```

## Command Reference

### test_inference.py

```
Usage: python test_inference.py [OPTIONS]

Options:
  --checkpoint PATH     Path to checkpoint file (default: pretrained_columns.pt)
  --mode MODE          Test mode: test, analyze, or compare (default: test)
  --prompt TEXT        Custom prompt for testing
  --max-length INT     Maximum generation length (default: 50)
  -h, --help          Show help message

Examples:
  # Analyze checkpoint
  python test_inference.py --mode analyze

  # Compare with custom prompt
  python test_inference.py --mode compare --prompt "What is Python?"

  # Full test with single prompt
  python test_inference.py --mode test --prompt "Hello"

  # Test with different checkpoint
  python test_inference.py --checkpoint old_checkpoint.pt --mode analyze
```

## Integration with Training Pipeline

### Step 1: Train
```bash
python pretraining.py --epochs 5 --batch-size 8 --dataset-size 5000
```
**Output:** `pretrained_columns.pt`

### Step 2: Quick Validation
```bash
python test_inference.py --mode analyze
```
**Checks:** File integrity, dimensions, training time

### Step 3: Detailed Testing
```bash
python test_inference.py --mode compare --prompt "Test prompt"
```
**Checks:** Outputs are valid (not NaN)

### Step 4: Full Validation
```bash
python test_inference.py --mode test
```
**Checks:** End-to-end inference pipeline

### Step 5: Proceed to Specialization
```bash
python uncertain.py  # Use uncertain-driven learning
# Or
python main.py      # Use full expandable model
```

## Performance Benchmarks

| Training Config | Time | Checkpoint Size | NaN Risk |
|----------------|------|----------------|----------|
| 1 epoch, 10 samples | 3 min | 111 MB | High |
| 3 epochs, 1000 samples | 15 min | 111 MB | Medium |
| 5 epochs, 5000 samples | 45 min | 111 MB | Low |
| 10 epochs, 10000 samples | 2 hours | 111 MB | Very Low |

**Note:** Checkpoint size stays the same (111 MB) regardless of training data - it's the model weights, not the dataset.

## Next Steps After Successful Testing

Once `test_inference.py` shows valid outputs:

1. **Use in main.py:**
   ```python
   model = ExpandableModelPNN()
   model.load_state()  # Loads pretrained columns
   ```

2. **Run uncertainty-driven learning:**
   ```bash
   python uncertain.py
   ```

3. **Deploy for interactive use:**
   - Columns will specialize on uncertainty patterns
   - Progressive learning without catastrophic forgetting

## Related Files

- `pretraining.py` - Train columns to create fertile ground
- `uncertain.py` - Uncertainty-driven specialization
- `main.py` - Full expandable model with memory
- `DATASET_MANAGEMENT.md` - Dataset caching guide
- `README.md` - Project overview

## Tips

1. **Always analyze first** - Quick validation before full testing
2. **Start small** - Test with 100 samples before training on 10k
3. **Monitor training** - Check loss values during training
4. **Compare outputs** - Ensure columns learn something
5. **Use caching** - Reuse datasets across training runs

---

**Remember:** Pretrained columns are meant to match the base model initially. Specialization happens during uncertainty-driven training, not pretraining!