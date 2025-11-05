#!/usr/bin/env python3
"""
Test Inference Script for Pretrained PNN Columns

This script tests the pretrained columns by:
1. Loading the checkpoint
2. Running inference with base model only
3. Running inference with base model + columns
4. Comparing outputs
"""

import torch
import torch.nn as nn
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


def create_column(hidden_size):
    """Create a single column with 3 layers (must match pretraining.py)"""
    return nn.Sequential(
        # Layer 1
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Dropout(0.1),
        # Layer 2
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Dropout(0.1),
        # Layer 3
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size)
    ).half()


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load pretrained columns from checkpoint"""
    print(f"\n{'='*60}")
    print(f"Loading Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"✓ Checkpoint loaded")
    print(f"  Hidden size: {checkpoint['hidden_size']}")
    print(f"  Vocab size: {checkpoint['vocab_size']}")
    print(f"  Training time: {checkpoint.get('training_time', 0) / 60:.2f} minutes")
    print(f"  Device: {checkpoint.get('device', 'unknown')}")

    return checkpoint


def initialize_model(checkpoint, model_name: str = "HuggingFaceTB/SmolLM-360M-Instruct", device: str = 'cuda'):
    """Initialize base model and columns from checkpoint"""
    print(f"\n{'='*60}")
    print(f"Initializing Model Components")
    print(f"{'='*60}")

    # Load base model
    print("\n1. Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )

    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False

    print(f"   ✓ Base model loaded ({sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params)")

    # Create columns
    print("\n2. Creating columns...")
    hidden_size = checkpoint['hidden_size']
    vocab_size = checkpoint['vocab_size']

    columns = nn.ModuleList([
        create_column(hidden_size) for _ in range(4)
    ])
    columns = columns.to(device).half()

    # Load column weights
    columns.load_state_dict(checkpoint['columns_state_dict'])
    print(f"   ✓ 4 columns loaded ({sum(p.numel() for p in columns[0].parameters()) / 1e6:.2f}M params each)")

    # Create output projection
    print("\n3. Creating output projection...")
    output_projection = nn.Linear(hidden_size, vocab_size).to(device).half()
    output_projection.load_state_dict(checkpoint['output_projection_state_dict'])
    print(f"   ✓ Output projection loaded")

    return base_model, tokenizer, columns, output_projection


def generate_base_only(base_model, tokenizer, prompt: str, max_length: int = 50, device: str = 'cuda'):
    """Generate using base model only"""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response


def generate_with_columns(base_model, tokenizer, columns, output_projection, prompt: str,
                          column_idx: int = 0, max_length: int = 50, device: str = 'cuda'):
    """Generate using base model + specific column"""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        # Get base hidden states
        base_outputs = base_model(**inputs, output_hidden_states=True)
        base_hidden = base_outputs.hidden_states[-1]

        # Pass through column
        column_hidden = columns[column_idx](base_hidden)

        # Get logits
        column_logits = output_projection(column_hidden)

        # Sample from column distribution
        probs = torch.softmax(column_logits[0, -1, :], dim=-1)
        next_token_id = torch.multinomial(probs, 1)

        # For simplicity, just return the first token prediction
        # (Full autoregressive generation would require a loop)
        next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)

    return next_token


def test_inference(checkpoint_path: str, test_prompts: list = None, max_length: int = 50):
    """Run inference tests"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Default test prompts
    if test_prompts is None:
        test_prompts = [
            "What is the capital of France?",
            "Write a Python function to calculate factorial",
            "Explain quantum computing in simple terms",
            "What is 15 + 27?",
            "Tell me about neural networks"
        ]

    # Load checkpoint and initialize model
    checkpoint = load_checkpoint(checkpoint_path, device)
    base_model, tokenizer, columns, output_projection = initialize_model(checkpoint, device=device)

    print(f"\n{'='*60}")
    print(f"Running Inference Tests")
    print(f"{'='*60}")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─'*60}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"{'─'*60}")
        print(f"Prompt: {prompt}")
        print()

        # Base model generation
        print("🔹 Base Model Only:")
        base_response = generate_base_only(base_model, tokenizer, prompt, max_length, device)
        print(f"   {base_response}")

        # Column-enhanced next token prediction
        print("\n🔸 Column-Enhanced (Next Token Only):")
        for col_idx in range(4):
            next_token = generate_with_columns(
                base_model, tokenizer, columns, output_projection,
                prompt, col_idx, max_length, device
            )
            print(f"   Column {col_idx}: '{next_token}'")

    print(f"\n{'='*60}")
    print(f"Inference Tests Complete")
    print(f"{'='*60}")


def analyze_checkpoint(checkpoint_path: str):
    """Analyze checkpoint contents without running inference"""
    print(f"\n{'='*60}")
    print(f"Checkpoint Analysis")
    print(f"{'='*60}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\n📊 Checkpoint Contents:")
    print(f"   Keys: {list(checkpoint.keys())}")
    print(f"\n📐 Model Dimensions:")
    print(f"   Hidden size: {checkpoint['hidden_size']}")
    print(f"   Vocabulary size: {checkpoint['vocab_size']}")
    print(f"\n⏱️  Training Info:")
    print(f"   Training time: {checkpoint.get('training_time', 0) / 60:.2f} minutes")
    print(f"   Device: {checkpoint.get('device', 'unknown')}")

    # Analyze columns
    columns_state = checkpoint['columns_state_dict']
    print(f"\n🏗️  Column Architecture:")
    num_columns = len([k for k in columns_state.keys() if k.startswith('0.')])
    print(f"   Number of columns: 4")
    print(f"   Parameters per column: {sum(p.numel() for p in columns_state.values()) / 4 / 1e6:.2f}M")

    # Analyze output projection
    output_state = checkpoint['output_projection_state_dict']
    print(f"\n📤 Output Projection:")
    print(f"   Parameters: {sum(p.numel() for p in output_state.values()) / 1e6:.2f}M")

    print(f"\n💾 File Size:")
    size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    print(f"   {size_mb:.2f} MB")


def compare_models(checkpoint_path: str, prompt: str = "What is artificial intelligence?"):
    """Compare base model vs column outputs side-by-side"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = load_checkpoint(checkpoint_path, device)
    base_model, tokenizer, columns, output_projection = initialize_model(checkpoint, device=device)

    print(f"\n{'='*60}")
    print(f"Model Comparison")
    print(f"{'='*60}")
    print(f"\nPrompt: {prompt}\n")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        # Base model
        base_outputs = base_model(**inputs, output_hidden_states=True)
        base_hidden = base_outputs.hidden_states[-1]
        base_logits = base_outputs.logits

        print("🔹 Base Model Logits:")
        print(f"   Shape: {base_logits.shape}")
        print(f"   Top 5 tokens:")
        top_base = torch.topk(base_logits[0, -1, :], 5)
        for prob, idx in zip(top_base.values, top_base.indices):
            token = tokenizer.decode([idx])
            print(f"      {prob:.4f} → '{token}'")

        # Each column
        for col_idx in range(4):
            print(f"\n🔸 Column {col_idx} Logits:")
            column_hidden = columns[col_idx](base_hidden)
            column_logits = output_projection(column_hidden)
            print(f"   Shape: {column_logits.shape}")
            print(f"   Top 5 tokens:")
            top_col = torch.topk(column_logits[0, -1, :], 5)
            for prob, idx in zip(top_col.values, top_col.indices):
                token = tokenizer.decode([idx])
                print(f"      {prob:.4f} → '{token}'")


def main():
    parser = argparse.ArgumentParser(description='Test pretrained PNN columns')
    parser.add_argument('--checkpoint', type=str, default='pretrained_columns.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'analyze', 'compare'],
                       help='Test mode: test (full inference), analyze (checkpoint info), compare (side-by-side)')
    parser.add_argument('--prompt', type=str,
                       help='Custom prompt for testing')
    parser.add_argument('--max-length', type=int, default=50,
                       help='Maximum generation length')

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"PNN Column Inference Test")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint}")

    if args.mode == 'analyze':
        analyze_checkpoint(args.checkpoint)

    elif args.mode == 'compare':
        prompt = args.prompt or "What is artificial intelligence?"
        compare_models(args.checkpoint, prompt)

    elif args.mode == 'test':
        if args.prompt:
            test_prompts = [args.prompt]
        else:
            test_prompts = None
        test_inference(args.checkpoint, test_prompts, args.max_length)


if __name__ == "__main__":
    main()