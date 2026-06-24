#!/usr/bin/env python3
"""
Evaluation script for H7: Uncertainty Window Optimal Range

Tests how different lower/upper entropy thresholds (uncertainty windows) affect:
1. U_ratio: % tokens where p is within [lower_upper] range
2. FP: % tokens with p < lower_th but routed (false positives)
"""

import torch
import torch.nn.functional as F
from uncertain import UncertaintyPNN
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import json
from pathlib import Path

# Test sentences with mixed uncertainty characteristics
TEST_SENTENCES = [
    # Confident statements (entropy low)
    "Paris is the capital of France.",
    "The sky is blue on a sunny day.",
    "2 + 2 = 4 mathematically.",
    
    # Ambiguous statements (entropy medium)
    "What time is it in Tokyo?",
    "How do I bake a chocolate cake?",
    "Please explain quantum computing.",
    
    # Highly uncertain statements (entropy high)
    "What is consciousness?",
    "Is there life beyond Earth?",
    "What is the meaning of existence?",
    "Why did the stock market drop?",
]


class WindowEvaluator:
    """Evaluates uncertainty window performance"""
    
    def __init__(self, test_sentences, num_windows=25):
        self.test_sentences = test_sentences
        # Generate window combinations covering the range [0.1, 0.9] × [0.5, 0.9]
        lower_bounds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        upper_bounds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        
        self.windows = []
        for lower in lower_bounds:
            for upper in upper_bounds:
                if upper > lower:
                    self.windows.append((lower, upper))
        
        # Sort by width for analysis
        self.windows.sort(key=lambda w: w[1] - w[0])
        
        # Track metrics per window
        self.metrics = {}
    
    def evaluate_all(self):
        """Evaluate all window combinations"""
        print(f"Testing {len(self.windows)} window combinations on {len(self.test_sentences)} test sentences\n")
        print("=" * 80)
        
        for lower, upper in self.windows:
            self.metrics[(lower, upper)] = self.evaluate_window(lower, upper)
        
        # Print results
        self.print_summary()
        
        # Save detailed results
        self.save_results()
    
    def evaluate_window(self, lower, upper):
        """Evaluate a specific window"""
        print(f"\n--- Window [{lower}, {upper}] ---")
        
        total_tokens = 0
        total_in_range = 0
        total_under_lower = 0
        
        for sentence in self.test_sentences:
            # Tokenize and get logits
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                max_length=128
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            with torch.no_grad():
                base_out = UncertaintyPNN._get_logits_without_init(self.base_model, inputs)
                logits = base_out.logits
            
            # Detect tokens in uncertainty window
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            
            # Check routing decision based on actual detection logic
            uncertain_mask = (max_probs > lower) & (max_probs < upper)
            
            total_tokens += len(max_probs[0])
            total_in_range += uncertain_mask.sum().item()
            total_under_lower += ((max_probs[0] > lower).sum() & (~uncertain_mask)).sum().item()
            
            if (lower, upper) == (0.3, 0.7):  # Report sample for the default
                print(f"\n  Sentence: '{sentence}'")
                print(f"  Total tokens: {len(max_probs[0])}")
                print(f"  Tokens in window: {uncertain_mask.sum().item()}")
                print(f"  Probabilities: {[f'{p:.3f}' for p in max_probs[0][:8]]}")
        
        # Calculate metrics
        u_ratio = total_in_range / total_tokens if total_tokens > 0 else 0.0
        fp = total_under_lower / total_tokens if total_tokens > 0 else 0.0
        
        result = {
            'lower': lower,
            'upper': upper,
            'u_ratio': u_ratio,
            'fp': fp,
            'tokens_in_window': total_in_range,
            'tokens_total': total_tokens,
            'tokens_under_lower': total_under_lower
        }
        
        print(f"\n  Summary:")
        print(f"    Tokens in window: {total_in_range}/{total_tokens} ({u_ratio*100:.2f}%)")
        print(f"    False positives (p < lower): {total_under_lower}/{total_tokens} ({fp*100:.2f}%)")
        
        return result
    
    @staticmethod
    def _get_logits_without_init(base_model_name, inputs):
        """Helper: Get logits without initializing UncertaintyPNN (requires mocking)"""
        # For simplicity in this test, we'll use UncertaintyPNN's internal detection
        # but this is a placeholder for the actual model
        model = UncertaintyPNN(base_model_name=base_model_name)
        return model._get_logits_base(inputs)
    
    def print_summary(self):
        """Print top/bottom performing windows"""
        print("\n" + "=" * 80)
        print("SUMMARY: Window Performance")
        print("=" * 80)
        
        # Sort by U_ratio
        sorted_by_uratio = sorted(self.metrics.items(), key=lambda x: x[1]['u_ratio'], reverse=True)
        
        print("\n--- Windows with highest U_ratio (good coverage) ---")
        for i, ((lower, upper), metrics) in enumerate(sorted_by_uratio[:5]):
            print(f"{i+1}. [{lower}, {upper}]: U_ratio={metrics['u_ratio']*100:.2f}%, FP={metrics['fp']*100:.2f}%")
        
        # Sort by lowest FP (good precision)
        sorted_by_fp = sorted(self.metrics.items(), key=lambda x: x[1]['fp'])
        
        print("\n--- Windows with lowest FP (good precision) ---")
        for i, ((lower, upper), metrics) in enumerate(sorted_by_fp[:5]):
            print(f"{i+1}. [{lower}, {upper}]: U_ratio={metrics['u_ratio']*100:.2f}%, FP={metrics['fp']*100:.2f}%")
    
    def save_results(self):
        """Save detailed results to JSON"""
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        output_path = results_dir / 'uncertainty_window_eval_results.json'
        
        with open(output_path, 'w') as f:
            json.dump({
                'eval_id': 'H7_uncertainty_window',
                'n_windows': len(self.windows),
                'n_sentences': len(self.test_sentences),
                'windows': self.metrics
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    print("=" * 80)
    print("EVALUATING UNCERTAINTY WINDOW RANGE (H7)")
    print("=" * 80)
    print("\nHypothesis: The uncertainty window [0.25, 0.75] maximizes")
    print("token-level uncertainty coverage while minimizing false positives")
    
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
    
    evaluator = WindowEvaluator(TEST_SENTENCES)
    evaluator.evaluate_all()
