#!/usr/bin/env python3
"""
Diagnostic to test a single training step and identify where NaN originates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_column(hidden_size):
    """Create a single column with 3 layers"""
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Dropout(0.1),
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size)
    ).half()

def test_single_step():
    """Test a single training step in detail."""

    print("="*60)
    print("SINGLE TRAINING STEP DIAGNOSTIC")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print("\n1. Loading base model...")
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.eval()

    for param in base_model.parameters():
        param.requires_grad = False

    # Create column and output projection
    print("\n2. Creating column and output projection...")
    hidden_size = base_model.config.hidden_size
    vocab_size = base_model.config.vocab_size

    column = create_column(hidden_size).to(device)
    output_projection = nn.Linear(hidden_size, vocab_size).to(device).half()

    # Initialize output projection from base model
    with torch.no_grad():
        output_projection.weight.copy_(base_model.lm_head.weight)
        if base_model.lm_head.bias is not None:
            output_projection.bias.copy_(base_model.lm_head.bias)

    # Freeze output projection
    for param in output_projection.parameters():
        param.requires_grad = False

    print(f"   Column parameters: {sum(p.numel() for p in column.parameters() if p.requires_grad):,}")
    print(f"   Output projection frozen: {not any(p.requires_grad for p in output_projection.parameters())}")

    # Create optimizer
    print("\n3. Creating optimizer...")
    optimizer = torch.optim.AdamW(column.parameters(), lr=1e-4)

    # Test with a simple prompt
    print("\n4. Testing with sample prompt...")
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors='pt', truncation=True, max_length=128).to(device)

    # Get base model outputs
    print("\n5. Getting base model outputs...")
    with torch.no_grad():
        base_outputs = base_model(**inputs, output_hidden_states=True)
        base_hidden = base_outputs.hidden_states[-1]
        base_logits = base_outputs.logits

    print(f"   Base hidden: {base_hidden.shape}, dtype={base_hidden.dtype}")
    print(f"   Base hidden has NaN: {torch.isnan(base_hidden).any().item()}")
    print(f"   Base hidden has Inf: {torch.isinf(base_hidden).any().item()}")
    print(f"   Base hidden min: {base_hidden.min().item():.4f}, max: {base_hidden.max().item():.4f}")

    # Column forward pass
    print("\n6. Column forward pass...")
    column_out = column(base_hidden)
    print(f"   Column out: {column_out.shape}, dtype={column_out.dtype}")
    print(f"   Column out has NaN: {torch.isnan(column_out).any().item()}")
    print(f"   Column out has Inf: {torch.isinf(column_out).any().item()}")
    print(f"   Column out min: {column_out.min().item():.4f}, max: {column_out.max().item():.4f}")

    # Output projection
    print("\n7. Output projection...")
    column_logits = output_projection(column_out)
    print(f"   Column logits: {column_logits.shape}, dtype={column_logits.dtype}")
    print(f"   Column logits has NaN: {torch.isnan(column_logits).any().item()}")
    print(f"   Column logits has Inf: {torch.isinf(column_logits).any().item()}")
    print(f"   Column logits min: {column_logits.min().item():.4f}, max: {column_logits.max().item():.4f}")

    # Compute KL divergence
    print("\n8. Computing KL divergence...")
    column_log_probs = F.log_softmax(column_logits.view(-1, column_logits.size(-1)), dim=-1)
    base_probs = F.softmax(base_logits.view(-1, base_logits.size(-1)), dim=-1)

    print(f"   Column log probs has NaN: {torch.isnan(column_log_probs).any().item()}")
    print(f"   Column log probs has Inf: {torch.isinf(column_log_probs).any().item()}")
    print(f"   Base probs has NaN: {torch.isnan(base_probs).any().item()}")
    print(f"   Base probs has Inf: {torch.isinf(base_probs).any().item()}")

    loss = F.kl_div(column_log_probs, base_probs, reduction='batchmean')
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Loss has NaN: {torch.isnan(loss).item()}")
    print(f"   Loss has Inf: {torch.isinf(loss).item()}")

    # Backward pass
    print("\n9. Backward pass...")
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    print("\n10. Checking gradients...")
    has_nan_grad = False
    max_grad = 0
    for name, param in column.named_parameters():
        if param.grad is not None:
            grad_nan = torch.isnan(param.grad).any().item()
            grad_inf = torch.isinf(param.grad).any().item()
            grad_max = param.grad.abs().max().item()
            max_grad = max(max_grad, grad_max)

            if grad_nan or grad_inf:
                print(f"   ✗ {name}: NaN={grad_nan}, Inf={grad_inf}, Max={grad_max:.4e}")
                has_nan_grad = True
            elif grad_max > 1000:
                print(f"   ! {name}: Max gradient very large: {grad_max:.4e}")

    print(f"   Max gradient across all params: {max_grad:.4e}")
    print(f"   Has NaN gradients: {has_nan_grad}")

    # Gradient clipping
    print("\n11. Gradient clipping...")
    grad_norm_before = torch.sqrt(sum(p.grad.pow(2).sum() for p in column.parameters() if p.grad is not None)).item()
    print(f"   Grad norm before clipping: {grad_norm_before:.4e}")

    grad_norm = torch.nn.utils.clip_grad_norm_(column.parameters(), 1.0)
    print(f"   Grad norm after clipping: {grad_norm:.4e}")

    # Check weights before optimizer step
    print("\n12. Weights before optimizer step...")
    weight_stats_before = {}
    for name, param in column.named_parameters():
        weight_stats_before[name] = {
            'min': param.data.min().item(),
            'max': param.data.max().item(),
            'mean': param.data.mean().item(),
            'nan': torch.isnan(param.data).any().item(),
            'inf': torch.isinf(param.data).any().item()
        }
        if weight_stats_before[name]['nan'] or weight_stats_before[name]['inf']:
            print(f"   ✗ {name}: Already has NaN or Inf BEFORE optimizer step!")

    # Optimizer step
    print("\n13. Optimizer step...")
    optimizer.step()

    # Check weights after optimizer step
    print("\n14. Weights after optimizer step...")
    for name, param in column.named_parameters():
        stats_after = {
            'min': param.data.min().item(),
            'max': param.data.max().item(),
            'mean': param.data.mean().item(),
            'nan': torch.isnan(param.data).any().item(),
            'inf': torch.isinf(param.data).any().item()
        }

        if stats_after['nan'] or stats_after['inf']:
            print(f"   ✗ {name}: NaN or Inf detected!")
            print(f"      Before: min={weight_stats_before[name]['min']:.4e}, "
                  f"max={weight_stats_before[name]['max']:.4e}, "
                  f"mean={weight_stats_before[name]['mean']:.4e}")
            print(f"      After:  min={stats_after['min']:.4e}, "
                  f"max={stats_after['max']:.4e}, "
                  f"mean={stats_after['mean']:.4e}")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_single_step()