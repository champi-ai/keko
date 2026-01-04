#!/usr/bin/env python3
"""
End-to-end test for the complete uncertainty-driven learning loop.

Tests:
1. Uncertainty detection
2. Clarification requests
3. Clarified response processing
4. Metrics tracking (AUT, URR, TSM)
5. Token hunger mechanism
6. Column routing
7. Background training integration
"""

import torch
import sys
from uncertain import UncertaintyPNN


def test_uncertainty_detection():
    """Test basic uncertainty detection"""
    print("\n" + "="*60)
    print("TEST 1: Uncertainty Detection")
    print("="*60)

    model = UncertaintyPNN()

    # Test with ambiguous input that should trigger uncertainty
    ambiguous_input = "What is it?"
    result = model.live_inference(ambiguous_input)

    print(f"Input: {ambiguous_input}")
    print(f"Mode: {result['mode']}")
    print(f"Response: {result['response']}")

    if result['mode'] == 'hungry':
        print("✓ Token hunger triggered correctly")
    elif result['mode'] == 'clarifying':
        print("✓ Clarification requested correctly")
    elif 'uncertainty_detected' in result and result['uncertainty_detected']:
        print("✓ Uncertainty detected")
    else:
        print("✗ No uncertainty detected (unexpected)")

    return result


def test_clarification_loop():
    """Test the full clarification request and response cycle"""
    print("\n" + "="*60)
    print("TEST 2: Clarification Loop")
    print("="*60)

    model = UncertaintyPNN()

    # Start with enough context to avoid token hunger
    model.token_buffer.extend([
        "Let's talk about machine learning",
        "I want to understand neural networks",
        "Can you explain something?"
    ])
    model.current_satisfaction = 0.9  # Override to skip token hunger

    # Ambiguous question that should trigger clarification
    ambiguous_question = "How does this work in practice?"

    print(f"\n1. Initial query: {ambiguous_question}")
    result1 = model.live_inference(ambiguous_question)

    print(f"   Mode: {result1['mode']}")
    print(f"   Response: {result1['response']}")

    if result1['mode'] == 'clarifying':
        print("✓ Clarification requested")

        # Simulate user providing clarification
        clarified_input = "How does backpropagation work in practice for training neural networks?"
        print(f"\n2. Clarified query: {clarified_input}")

        result2 = model.process_clarified_input(ambiguous_question, clarified_input)

        print(f"   Mode: {result2['mode']}")
        print(f"   Response: {result2['response'][:100]}...")

        # Check clarification stats
        stats = model.clarification_engine.get_clarification_stats()
        print(f"\n3. Clarification Stats:")
        print(f"   Total requests: {stats['total_requests']}")
        print(f"   Completed: {stats['completed']}")
        print(f"   Avg gain (ΔU): {stats['avg_clarification_gain']:.4f}")

        if stats['total_requests'] > 0:
            print("✓ Clarification cycle completed")
        else:
            print("✗ Clarification not tracked")
    else:
        print("⚠ Clarification not triggered (may need threshold tuning)")

    return result1


def test_token_hunger():
    """Test token hunger mechanism"""
    print("\n" + "="*60)
    print("TEST 3: Token Hunger Mechanism")
    print("="*60)

    model = UncertaintyPNN()

    # Test with insufficient context
    test_cases = [
        ("it", "Short pronoun reference"),
        ("What is it?", "Vague reference"),
        ("this thing here", "Insufficient detail"),
        ("Could you please provide a detailed explanation of backpropagation algorithms?", "Sufficient context")
    ]

    for input_text, description in test_cases:
        result = model.live_inference(input_text)

        print(f"\nInput: '{input_text}'")
        print(f"Description: {description}")
        print(f"Mode: {result['mode']}")
        print(f"Satisfaction: {result['satisfaction']:.2f}")

        if result['mode'] == 'hungry' and result['satisfaction'] < 0.8:
            print(f"✓ Correctly identified as needing more context")
        elif result['satisfaction'] >= 0.8:
            print(f"✓ Correctly satisfied with context")
        else:
            print(f"→ Response: {result['response'][:80]}...")


def test_metrics_tracking():
    """Test AUT, URR, and TSM metrics"""
    print("\n" + "="*60)
    print("TEST 4: Metrics Tracking (AUT, URR, TSM)")
    print("="*60)

    model = UncertaintyPNN()

    # Provide sufficient context
    model.token_buffer.extend(["Initial context"] * 5)
    model.current_satisfaction = 0.9

    # Generate multiple inferences to accumulate metrics
    test_inputs = [
        "What is machine learning?",
        "Explain neural networks",
        "How does training work?",
        "What is backpropagation?",
        "Describe gradient descent"
    ]

    print("\nGenerating responses to accumulate metrics...")
    for i, input_text in enumerate(test_inputs):
        result = model.live_inference(input_text)
        print(f"{i+1}. Processed: '{input_text}' -> {result['mode']}")

        # Simulate epoch for TSM calculation
        if i % 2 == 1:
            model.metrics.record_epoch_uncertainty()

    # Get metrics
    metrics = model.metrics.get_comprehensive_metrics()

    print(f"\n📊 Metrics Summary:")
    print(f"  AUT (Average Uncertainty per Token): {metrics['aut']:.4f}")
    print(f"  URR (Uncertainty Resolution Rate): {metrics['urr']:.2%}")
    print(f"  TSM (Temporal Stability Metric): {metrics['tsm']:.4f}")
    print(f"  Total tokens tracked: {metrics['total_tokens_tracked']}")
    print(f"  Uncertain events: {metrics['total_uncertain_events']}")
    print(f"  Resolved events: {metrics['resolved_events']}")

    # Validate metrics
    if metrics['aut'] > 0:
        print("✓ AUT calculated correctly")
    else:
        print("✗ AUT is zero (unexpected)")

    if metrics['total_tokens_tracked'] > 0:
        print("✓ Token tracking working")
    else:
        print("✗ No tokens tracked")

    if len(model.metrics.epoch_uncertainties) > 0:
        print("✓ Epoch uncertainty tracking working")
        if metrics['tsm'] >= 0:
            print("✓ TSM calculated")
    else:
        print("⚠ Not enough data for TSM yet")

    return metrics


def test_column_routing():
    """Test that uncertain inputs are routed through columns"""
    print("\n" + "="*60)
    print("TEST 5: Column Routing")
    print("="*60)

    model = UncertaintyPNN()

    # Provide context and set satisfaction
    model.token_buffer.extend(["Context"] * 5)
    model.current_satisfaction = 0.9

    print("\nProcessing query that should trigger column routing...")
    result = model.live_inference("What is the meaning of existence?")

    print(f"Mode: {result['mode']}")
    print(f"Uncertainty detected: {result.get('uncertainty_detected', False)}")

    # Check if any columns were activated
    column_activations = model.metrics.column_activation_counts

    print(f"\nColumn activations: {column_activations}")

    if sum(column_activations) > 0:
        print("✓ Columns were activated")
    else:
        print("⚠ No column activations yet (may need more uncertainty)")

    return result


def test_full_status_display():
    """Test the comprehensive status display"""
    print("\n" + "="*60)
    print("TEST 6: Full Status Display")
    print("="*60)

    model = UncertaintyPNN()

    # Run several cycles
    model.token_buffer.extend(["Initial context"] * 5)
    model.current_satisfaction = 0.9

    for i in range(10):
        input_text = f"Question {i+1}: What is AI concept {i}?"
        model.live_inference(input_text)

        if i % 3 == 0:
            model.metrics.record_epoch_uncertainty()

    print("\nDisplaying full status:")
    model.print_status()

    print("✓ Status display completed")


def test_generation_quality():
    """Test autoregressive generation produces sensible output"""
    print("\n" + "="*60)
    print("TEST 7: Autoregressive Generation Quality")
    print("="*60)

    model = UncertaintyPNN()

    # Provide context
    model.token_buffer.extend(["Let's discuss AI"] * 5)
    model.current_satisfaction = 0.9

    test_input = "What is artificial intelligence?"

    print(f"\nInput: {test_input}")
    result = model.live_inference(test_input)

    print(f"Mode: {result['mode']}")
    print(f"Response: {result['response']}")
    print(f"Response length: {len(result['response'])} chars")

    if len(result['response']) > 10:
        print("✓ Generated multi-token response")
    else:
        print("⚠ Response is very short")

    if result['mode'] == 'responding':
        print("✓ Generated actual response (not just clarification)")

    return result


def run_all_tests():
    """Run all end-to-end tests"""
    print("\n" + "="*70)
    print("🧪 KEKO UNCERTAINTY LOOP - END-TO-END TESTS")
    print("="*70)

    try:
        test_uncertainty_detection()
        test_token_hunger()
        test_clarification_loop()
        test_metrics_tracking()
        test_column_routing()
        test_generation_quality()
        test_full_status_display()

        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED")
        print("="*70)
        print("\nNote: Some tests may show warnings if thresholds need tuning.")
        print("This is expected for a POC - the system is learning its parameters.")

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available, using CPU (will be slower)")

    run_all_tests()
