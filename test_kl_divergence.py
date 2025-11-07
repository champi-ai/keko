#!/usr/bin/env python3
"""
Diagnostic script to test KL divergence computation and identify NaN sources.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_kl_computation():
    """Test KL divergence computation with actual model outputs."""

    print("="*60)
    print("KL DIVERGENCE DIAGNOSTIC TEST")
    print("="*60)

    # Load model
    print("\n1. Loading base model...")
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.eval()

    device = base_model.device
    print(f"   Device: {device}")
    print(f"   Model dtype: {base_model.dtype}")

    # Test with a simple prompt
    print("\n2. Testing with sample prompt...")
    test_prompt = "Hello, how are you?"
    inputs = tokenizer(test_prompt, return_tensors='pt', truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = base_model(**inputs, output_hidden_states=True)
        base_hidden = outputs.hidden_states[-1]
        base_logits = outputs.logits

    print(f"   Input shape: {inputs.input_ids.shape}")
    print(f"   Hidden shape: {base_hidden.shape}")
    print(f"   Logits shape: {base_logits.shape}")
    print(f"   Logits dtype: {base_logits.dtype}")

    # Check for inf/nan in base outputs
    print("\n3. Checking base model outputs...")
    print(f"   Hidden has NaN: {torch.isnan(base_hidden).any().item()}")
    print(f"   Hidden has Inf: {torch.isinf(base_hidden).any().item()}")
    print(f"   Logits has NaN: {torch.isnan(base_logits).any().item()}")
    print(f"   Logits has Inf: {torch.isinf(base_logits).any().item()}")
    print(f"   Logits min: {base_logits.min().item():.4f}")
    print(f"   Logits max: {base_logits.max().item():.4f}")

    # Test KL divergence with same distribution (should be ~0)
    print("\n4. Testing KL divergence (same distribution)...")
    logits_flat = base_logits.view(-1, base_logits.size(-1))

    log_probs = F.log_softmax(logits_flat, dim=-1)
    probs = F.softmax(logits_flat, dim=-1)

    print(f"   Log probs has NaN: {torch.isnan(log_probs).any().item()}")
    print(f"   Log probs has Inf: {torch.isinf(log_probs).any().item()}")
    print(f"   Probs has NaN: {torch.isnan(probs).any().item()}")
    print(f"   Probs has Inf: {torch.isinf(probs).any().item()}")
    print(f"   Probs min: {probs.min().item():.6e}")
    print(f"   Probs max: {probs.max().item():.6e}")
    print(f"   Probs sum (should be ~1): {probs.sum(dim=-1).mean().item():.6f}")

    kl_div = F.kl_div(log_probs, probs, reduction='batchmean')
    print(f"   KL divergence (self): {kl_div.item():.6e}")
    print(f"   KL has NaN: {torch.isnan(kl_div).item()}")
    print(f"   KL has Inf: {torch.isinf(kl_div).item()}")

    # Test with slightly perturbed distribution
    print("\n5. Testing KL with perturbed distribution...")
    noise = torch.randn_like(base_logits) * 0.1
    perturbed_logits = base_logits + noise

    perturbed_log_probs = F.log_softmax(perturbed_logits.view(-1, perturbed_logits.size(-1)), dim=-1)

    print(f"   Perturbed log probs has NaN: {torch.isnan(perturbed_log_probs).any().item()}")
    print(f"   Perturbed log probs has Inf: {torch.isinf(perturbed_log_probs).any().item()}")

    kl_div_perturbed = F.kl_div(perturbed_log_probs, probs, reduction='batchmean')
    print(f"   KL divergence (perturbed): {kl_div_perturbed.item():.6e}")
    print(f"   KL has NaN: {torch.isnan(kl_div_perturbed).item()}")
    print(f"   KL has Inf: {torch.isinf(kl_div_perturbed).item()}")

    # Test with random logits (like untrained column)
    print("\n6. Testing KL with random logits (untrained column simulation)...")
    random_logits = torch.randn_like(base_logits) * 0.02  # Small initialization
    random_log_probs = F.log_softmax(random_logits.view(-1, random_logits.size(-1)), dim=-1)

    print(f"   Random logits min: {random_logits.min().item():.4f}")
    print(f"   Random logits max: {random_logits.max().item():.4f}")
    print(f"   Random log probs has NaN: {torch.isnan(random_log_probs).any().item()}")
    print(f"   Random log probs has Inf: {torch.isinf(random_log_probs).any().item()}")

    kl_div_random = F.kl_div(random_log_probs, probs, reduction='batchmean')
    print(f"   KL divergence (random): {kl_div_random.item():.6e}")
    print(f"   KL has NaN: {torch.isnan(kl_div_random).item()}")
    print(f"   KL has Inf: {torch.isinf(kl_div_random).item()}")

    # Test backward pass
    print("\n7. Testing backward pass...")
    kl_div_random.backward()
    print(f"   Backward pass completed without error")

    # Test with float32
    print("\n8. Testing with float32 precision...")
    base_logits_f32 = base_logits.float()
    random_logits_f32 = random_logits.float()

    log_probs_f32 = F.log_softmax(random_logits_f32.view(-1, random_logits_f32.size(-1)), dim=-1)
    probs_f32 = F.softmax(base_logits_f32.view(-1, base_logits_f32.size(-1)), dim=-1)

    kl_div_f32 = F.kl_div(log_probs_f32, probs_f32, reduction='batchmean')
    print(f"   KL divergence (float32): {kl_div_f32.item():.6e}")
    print(f"   KL has NaN: {torch.isnan(kl_div_f32).item()}")
    print(f"   KL has Inf: {torch.isinf(kl_div_f32).item()}")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_kl_computation()
