from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional
import os
import json
from multiprocessing import Manager, Process

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset_generator.uncertainty_prompts import IMPOSSIBLE_PROMPTS, CATEGORY_2_SELF_KNOWLEDGE,


@dataclass
class GenerationConfig:
    checkpoint_path: str
    prompts: Iterable[str] = field(default_factory=lambda: IMPOSSIBLE_PROMPTS)
    max_new_tokens: int = 1280
    temperature: float = 0.7
    top_k: int = 3
    repetition_penalty: float = 1.0
    num_samples: Optional[int] = 100
    device: Optional[str] = None
    output_path: str = 'datasets/keko_factset'
    save_json: bool = True
    save_jsonl: bool = True
    num_workers: int = 2
    append_mode: bool = False  # If True, append to existing dataset instead of overwriting


def _load_QuantizedGemma3_1B(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        torch_dtype=torch.bfloat16,        # Gemma 3 loves bfloat16
        device_map="auto",
        trust_remote_code=True,            # required for Gemma 3
    )
    model.eval()
    return tokenizer, model


def _capture_response(prompt: str, tokenizer, model, device, cfg: GenerationConfig):
    prompt_template = f"""<start_of_turn>user
    {prompt}<end_of_turn>
    <start_of_turn>model
    You are an honest AI. If you do not know the answer with certainty, you MUST say "I don’t know" or "I’m not sure" and briefly explain why.
    Answer:"""
    inputs = tokenizer(prompt_template, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 max_new_tokens=cfg.max_new_tokens,
                                 min_new_tokens=max(1, cfg.max_new_tokens // 8),
                                 temperature=0.1,
                                 top_k=1,
                                 repetition_penalty=cfg.repetition_penalty,
                                 do_sample=True,
                                 no_repeat_ngram_size=3,
                                 pad_token_id=tokenizer.eos_token_id)
        logits = model(outputs, output_hidden_states=False).logits

    raw_response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    response = raw_response.split('\n')[0].strip()
    if not response:
        response = raw_response

    probs = F.softmax(logits[:, -1, :], dim=-1)
    max_probs = probs.max(dim=-1).values
    uncertainty = 1.0 - max_probs
    token_ids = outputs[0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    trimmed_encoding = tokenizer(response, add_special_tokens=False, return_tensors='pt')
    trimmed_len = trimmed_encoding['input_ids'].shape[-1]
    record = {
        'prompt': prompt,
        'response': response,
        'mean_confidence': float(max_probs.mean().item()),
        'mean_uncertainty': float(uncertainty.mean().item()),
        'uncertainty_ratio': float(((max_probs < 0.7) & (max_probs > 0.3)).float().mean().item()),
        'tokens': tokens[:trimmed_len],
        'token_confidences': [float(p) for p in max_probs.tolist()[:trimmed_len]],
        'created_at': datetime.now().isoformat()
    }
    _record = record.copy()
    _record.pop('token_confidences')
    _record.pop('tokens')
    print(json.dumps(_record, indent=2))
    return record


def _worker_chunk(worker_id: int, prompts: List[str], cfg: GenerationConfig, queue):
    device = torch.device(cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tokenizer, model = _load_QuantizedGemma3_1B(device)
    for prompt in prompts:
        record = _capture_response(prompt, tokenizer, model, device, cfg)
        record['worker_id'] = worker_id
        queue.put(record)


def generate_factset(cfg: GenerationConfig):
    cfg.device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    prompts = list(cfg.prompts)
    if cfg.num_samples:
        prompts = prompts[: cfg.num_samples]

    if not prompts:
        raise ValueError('No prompts supplied for dataset generation.')

    # Load existing dataset if in append mode
    existing_records = []
    if cfg.append_mode and os.path.exists(os.path.join(cfg.output_path, 'data-00000-of-00001.arrow')):
        try:
            existing_dataset = Dataset.load_from_disk(cfg.output_path)
            existing_records = list(existing_dataset)
            print(f"📎 Append mode: Loaded {len(existing_records)} existing records from {cfg.output_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing dataset for append: {e}")
            print("   Creating new dataset instead.")

    manager = Manager()
    queue = manager.Queue()
    chunk_size = max(1, (len(prompts) + cfg.num_workers - 1) // cfg.num_workers)

    processes: List[Process] = []
    for worker_id, idx in enumerate(range(0, len(prompts), chunk_size)):
        chunk = prompts[idx : idx + chunk_size]
        p = Process(target=_worker_chunk, args=(worker_id, chunk, cfg, queue))
        p.start()
        processes.append(p)

    records: List[dict] = []
    for _ in range(len(prompts)):
        records.append(queue.get())

    for p in processes:
        p.join()

    # Combine existing records with new ones in append mode
    if cfg.append_mode and existing_records:
        all_records = existing_records + records
        print(f"✅ Combined {len(existing_records)} existing + {len(records)} new = {len(all_records)} total records")
        dataset = Dataset.from_list(all_records)
    else:
        dataset = Dataset.from_list(records)

    dataset.save_to_disk(cfg.output_path)
    os.makedirs(cfg.output_path, exist_ok=True)
    if cfg.save_json:
        dataset.to_json(os.path.join(cfg.output_path, 'dataset.json'))
    if cfg.save_jsonl:
        dataset.to_json(os.path.join(cfg.output_path, 'dataset.jsonl'), lines=True)

    return cfg.output_path
