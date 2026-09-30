"""Shared fixtures: a tiny random-weight Llama and an offline word-level tokenizer."""

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def tiny_model():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    words = ("<unk> <pad> <eos> what is the capital of france paris 2 + 3 5 who wrote i don't know "
             "question answer : ? . a b c d e f g h").split()
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                                  eos_token="<eos>")
    tok.padding_side = "left"
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, pad_token_id=vocab["<pad>"],
                      eos_token_id=vocab["<eos>"], bos_token_id=vocab["<eos>"])
    model = LlamaForCausalLM(cfg).eval()
    return model, tok


@pytest.fixture
def rng():
    return np.random.default_rng(0)
