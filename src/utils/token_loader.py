"""token_loader.py — Dataset and DataLoader utilities for PaES-LLM experiments."""

from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TextCorpusDataset(Dataset):
    """Tokenised text dataset with fixed-length padding."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


def _build_loader(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    shuffle: bool = False,
) -> DataLoader:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return DataLoader(
        TextCorpusDataset(texts, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def get_token_loader(
    texts: List[str],
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    batch_size: int = 1,
    max_length: int = 128,
) -> DataLoader:
    """DataLoader for an arbitrary list of strings."""
    return _build_loader(texts, model_name, batch_size, max_length)


def get_wikitext2_loader(
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    split: str = "test",
    n_samples: int = 200,
    max_length: int = 128,
    batch_size: int = 4,
) -> DataLoader:
    """DataLoader for the WikiText-2 benchmark (perplexity evaluation)."""
    try:
        from datasets import load_dataset
        ds    = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:n_samples]
    except Exception as exc:
        print(f"WikiText-2 unavailable ({exc}). Using fallback corpus.")
        texts = _fallback_corpus()
    return _build_loader(texts, model_name, batch_size, max_length)


def get_lambada_loader(
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    n_samples: int = 200,
    max_length: int = 128,
    batch_size: int = 4,
) -> DataLoader:
    """DataLoader for the LAMBADA benchmark (last-word prediction accuracy)."""
    try:
        from datasets import load_dataset
        ds    = load_dataset("EleutherAI/lambada_openai", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 30][:n_samples]
    except Exception as exc:
        print(f"LAMBADA unavailable ({exc}). Using fallback corpus.")
        texts = _fallback_corpus()
    return _build_loader(texts, model_name, batch_size, max_length)


def _fallback_corpus() -> List[str]:
    return [
        "Paraconsistent logic handles contradictory information without trivialization.",
        "Large Language Models consume massive electrical energy for global inference.",
        "Quantum decoherence in superconducting qubits requires cryogenic cooling.",
        "Epistemic uncertainty is modeled via Belnap L4 lattice structures.",
        "The transformer architecture relies on self-attention for sequence modeling.",
        "Energy-efficient inference is critical for sustainable AI deployment.",
        "Sparse computation methods reduce the computational cost of neural networks.",
        "Semantic preservation is essential when pruning attention in language models.",
    ]
