import argparse
import torch

torch.set_default_dtype(torch.float32)

import torch.nn.functional as F
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from test_inference import create_column


def load_columns_and_projection(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_size = checkpoint['hidden_size']
    vocab_size = checkpoint['vocab_size']

    columns = nn.ModuleList([create_column(hidden_size) for _ in range(4)])
    columns.load_state_dict(checkpoint['columns_state_dict'])
    columns = columns.to(device)

    output_projection = nn.Linear(hidden_size, vocab_size, ).to(device)
    output_projection.load_state_dict(checkpoint['output_projection_state_dict'])

    return columns, output_projection


def analyze_logits(logits: torch.Tensor, tokens_ids, tokenizer: AutoTokenizer):
    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1).values
    uncertainty = 1.0 - max_probs
    tokens = tokenizer.convert_ids_to_tokens(tokens_ids[0])
    return tokens, max_probs.squeeze(0), uncertainty.squeeze(0)


def run_probe(prompt: str, base_model, tokenizer, columns, output_projection, device, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = base_model(generated, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits

    response = tokenizer.decode(
        generated[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    tokens, max_probs, uncertainty = analyze_logits(logits, generated, tokenizer)
    avg_uncertainty = uncertainty.mean().item()
    uncertain_pct = ((max_probs < 0.7) & (max_probs > 0.3)).float().mean().item()

    column_stats = []
    for idx, column in enumerate(columns):
        column_hidden = column(hidden_states.float())
        column_logits = output_projection(column_hidden)
        column_probs = F.softmax(column_logits, dim=-1).max(dim=-1).values
        column_stats.append(1.0 - column_probs.mean().item())

    print("=" * 60)
    print(f"Prompt: {prompt!r}")
    print(f"Response: {response.strip()}")
    print(f"Avg uncertainty: {avg_uncertainty:.4f}, uncertain tokens: {uncertain_pct:.1%}")
    print("Column mean uncertainty:", ", ".join(f"Col{idx}:{val:.4f}" for idx, val in enumerate(column_stats)))
    print("-" * 60)

    header = "{:4s} {:>8s} {:>10s}  {}".format("idx", "token", "max_p", "uncertainty")
    print(header)
    print("-" * len(header))
    for idx, (token, max_p, unc) in enumerate(zip(tokens, max_probs.tolist(), uncertainty.tolist())):
        marker = "*" if 0.3 < max_p < 0.7 else " "
        print(f"{idx:4d} {token[:10]:>8s} {max_p:10.4f} {unc:10.4f} {marker}")

    print("=" * 60)


def main():
    # parser = argparse.ArgumentParser(description="Probe pretrained columns for uncertainty.")
    # parser.add_argument("--checkpoint", default="pretrained_columns.pt", help="Path to pretrained columns checkpoint")
    # parser.add_argument("--prompt", default="Explain how uncertainty emerges in reasoning.", help="Base prompt")
    # parser.add_argument("--max-tokens", type=int, default=64, help="Max new tokens to generate")
    # args = parser.parse_args()
    _checkpoint="pretrained_columns.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-360M-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-360M-Instruct",
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    base_model.eval()
    base_model.to(device)

    columns, output_projection = load_columns_and_projection(_checkpoint, device)
    output_projection.to(device)

    # 20 prompts designed to be maximally ambiguous (should trigger high uncertainty)
    _ambiguous_prompts = [
        # Category 1: Pure Ambiguity (Context Missing)
        "How to fix this",
        "Is it better?",
        "What should I use instead?",
        "Can you explain the problem?",
        "Which one is correct?",

        # Category 2: Pronoun Hell (Reference Missing)
        "Why did it fail?",
        "Should I change that?",
        "Is this the right approach?",
        "How do I make it work?",
        "What does he mean by that?",

        # Category 3: Incomplete Technical Questions
        "How do I install it?",
        "What's the syntax for this?",
        "Which version should I use?",
        "How do I configure the settings?",
        "What's causing the error?",

        # Category 4: Context-Dependent Comparisons
        "Which is faster?",
        "Is this more secure?",
        "What's the best option here?",
        "Should I optimize for this or that?",
        "Is there a better way to do it?"
    ]
    _non_ambiguous_prompts = [
        "What is 2 + 2?",
        # "Complete this sentence: The capital of France is",
        # "What color is the sky on a clear day?",
        # "How many days are in a week?",
        # "What is the chemical formula for water?",
        # "True or False: The Earth orbits the Sun.",
        # "What is the freezing point of water in Celsius?",
        # "How many letters are in the English alphabet?",
        # "What comes after Monday?",
        # "Convert 100 centimeters to meters:",
        # "What is the opposite of hot?",
        # "How many sides does a triangle have?",
        # "Complete: A, B, C, D,",
        # "What is 10 divided by 2?",
        # "The sun rises in the east and sets in the",
        # "How many minutes are in one hour?",
        # "What is the largest planet in our solar system?",
        # "Spell the word 'cat':",
        # "What year comes after 2024?",
        # "Is fire hot or cold?"
    ]
    _max_tokens = 3200
    for _prompt in _non_ambiguous_prompts:
        run_probe(
            _prompt,
            base_model,
            tokenizer,
            columns,
            output_projection,
            device,
            _max_tokens
        )


if __name__ == "__main__":
    main()
