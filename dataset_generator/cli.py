import argparse
from pathlib import Path

from dataset_generator.generation import GenerationConfig, generate_factset
from dataset_generator.uncertainty_prompts import IMPOSSIBLE_PROMPTS



def load_prompts(path: Path):
    if not path:
        return list()

    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate fact-response dataset from the frozen base model.")
    parser.add_argument("--output-path", type=Path, default=Path("datasets/keko_factset"), help="Directory to save the HuggingFace dataset")
    parser.add_argument("--prompts-file", type=Path, help="Optional newline-delimited prompts file")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Generation length per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--num-samples", type=int, default=10, help="Limit number of prompts")
    parser.add_argument("--save-json", action="store_true", default=True, help="Also export dataset as JSON")
    parser.add_argument("--save-jsonl", action="store_true", default=True, help="Also export dataset as newline-delimited JSONL")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel worker processes")
    parser.add_argument("--append", action="store_true", help="Append to existing dataset instead of overwriting")
    args = parser.parse_args()

    prompts = list(IMPOSSIBLE_PROMPTS)

    # Load custom prompts from file if provided
    if args.prompts_file:
        custom_prompts = load_prompts(args.prompts_file)
        if custom_prompts:
            prompts = custom_prompts
            print(f"📋 Loaded {len(prompts)} prompts from {args.prompts_file}")

    cfg = GenerationConfig(
        checkpoint_path="pretrained_columns.pt",
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        num_samples=args.num_samples,
        device=args.device,
        output_path=str(args.output_path),
        save_json=args.save_json,
        save_jsonl=args.save_jsonl,
        num_workers=args.num_workers,
        append_mode=args.append
    )

    output = generate_factset(cfg)
    print(f"Dataset saved to {output}")


if __name__ == "__main__":
    main()
