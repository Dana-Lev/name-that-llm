"""Features read out of a frozen open language model (for example Qwen 3).

The model is not trained. Each passage is run through it once and we keep:

- "stats": 14 numbers on how predictable the passage is to the model
  (log-probability of each actual next token, entropy of the model's
  prediction, rank of the actual token among the model's guesses).
- "L<k>": the hidden vectors of layer k averaged over the passage, for four
  layers spread through the network.

This is heavy: run it on a GPU (Kaggle), not on a laptop.
"""

import json
import os
from pathlib import Path

import numpy as np

STAT_NAMES = [
    "logp_mean", "logp_std", "logp_q05", "logp_q25", "logp_q50", "logp_q75", "logp_q95",
    "entropy_mean", "entropy_std", "logp_plus_entropy_mean",
    "top1_share", "top5_share", "top20_share", "log_rank_mean",
]  # fmt: skip

_loaded = {}  # source -> (tokenizer, model, device), so the model loads once


def find_llm(root="/kaggle/input"):
    """Folder of the attached causal language model.

    $NTL_LLM_DIR wins. Otherwise the first folder under `root` whose
    config.json describes a ...ForCausalLM model and that has weights.
    """
    override = os.environ.get("NTL_LLM_DIR")
    if override:
        return override
    for config in sorted(Path(root).rglob("config.json")):
        folder = config.parent
        try:
            architectures = json.loads(config.read_text()).get("architectures") or []
        except (json.JSONDecodeError, OSError):
            continue
        has_weights = any(folder.glob("*.safetensors")) or (folder / "pytorch_model.bin").exists()
        if has_weights and any(a.endswith("ForCausalLM") for a in architectures):
            return str(folder)
    raise FileNotFoundError(f"no causal language model under {root}: attach one (e.g. Qwen 3) as an input")


def _load(source):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if source not in _loaded:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        half = torch.float32 if device == "cpu" else torch.float16
        try:  # the argument was renamed from torch_dtype to dtype in newer transformers
            model = AutoModelForCausalLM.from_pretrained(source, dtype=half)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=half)
        model = model.to(device).eval()
        _loaded[source] = (tokenizer, model, device)
    return _loaded[source]


def pick_layers(n_layers):
    return sorted({n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers})


def extract(texts, source=None, batch_size=8, max_length=512, verbose=True):
    """Return {"stats": (n, 14), "L<k>": (n, hidden_size), ...} for the passages."""
    import torch

    texts = list(texts)
    source = find_llm() if source is None else source
    tokenizer, model, device = _load(source)
    layers = pick_layers(model.config.num_hidden_layers)
    features = {f"L{k}": np.zeros((len(texts), model.config.hidden_size), dtype=np.float32) for k in layers}
    features["stats"] = np.zeros((len(texts), len(STAT_NAMES)), dtype=np.float32)

    order = np.argsort([len(text) for text in texts], kind="stable")
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            # An empty passage would give an empty token row; one space keeps it valid.
            batch = [texts[i] if texts[i].strip() else " " for i in rows]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            out = model(**encoded, output_hidden_states=True)
            mask = encoded["attention_mask"]
            weights = mask.unsqueeze(-1).float()
            for k in layers:
                hidden = out.hidden_states[k].float()
                features[f"L{k}"][rows] = ((hidden * weights).sum(1) / weights.sum(1)).cpu().numpy()

            # Position t predicts token t+1. One passage at a time: the scores
            # over the whole vocabulary (150k words per token) for a full batch
            # don't fit in a 16 GB GPU.
            lengths = mask.sum(1).tolist()
            for j, i in enumerate(rows):
                n = int(lengths[j])
                if n < 2:  # a one-token passage has nothing to predict
                    continue
                log_probs = torch.log_softmax(out.logits[j, : n - 1].float(), dim=-1)
                target = encoded["input_ids"][j, 1:n].unsqueeze(-1)
                actual = log_probs.gather(-1, target)
                lp = actual.squeeze(-1).cpu().numpy()
                en = (-(log_probs.exp() * log_probs).sum(-1)).cpu().numpy()
                rk = (log_probs > actual).sum(-1).cpu().numpy()
                del log_probs
                features["stats"][i] = [
                    lp.mean(), lp.std(), *np.quantile(lp, [0.05, 0.25, 0.5, 0.75, 0.95]),
                    en.mean(), en.std(), (lp + en).mean(),
                    (rk == 0).mean(), (rk < 5).mean(), (rk < 20).mean(), np.log1p(rk).mean(),
                ]  # fmt: skip
            if verbose and (start // batch_size) % 50 == 0:
                print(f"llm features: {start}/{len(texts)} passages", flush=True)
    return features
