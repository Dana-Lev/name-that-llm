"""Fine-tune a pretrained text encoder (DeBERTa-v3) to name the model.

The encoder reads the passage as tokens, we average its output vectors over
the passage and put one linear layer with 18 outputs on top. All weights are
trained, with a small learning rate for the pretrained part.

torch and transformers are imported inside the class so the rest of the
project works without them.
"""

import os
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "microsoft/deberta-v3-base"


def find_pretrained(name=DEFAULT_MODEL):
    """Where to load the pretrained weights from.

    Kaggle notebooks run without internet, so there the weights must be
    attached as an input; we look for a folder under /kaggle/input whose path
    contains the model's short name. Elsewhere the Hugging Face name is used
    (downloaded once, then cached). $NTL_PRETRAINED_DIR overrides both.
    """
    override = os.environ.get("NTL_PRETRAINED_DIR")
    if override:
        return override
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        return find_weights_folder(kaggle_input, name)
    return name


def _squash(text):
    """Lower-case letters and digits only, so 'DeBERTa-v3-base' matches 'debertav3base'."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def find_weights_folder(root, name=DEFAULT_MODEL):
    """The folder under `root` that holds the pretrained model `name`.

    It must have a config.json next to a weights file, and its path must
    contain the model's short name (ignoring case and punctuation). An input
    that is merely called deberta-v3-base but has no weights doesn't count.
    """
    short = name.split("/")[-1]
    for config in sorted(Path(root).rglob("config.json")):
        folder = config.parent
        has_weights = any((folder / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
        if has_weights and _squash(short) in _squash(str(folder)):
            return str(folder)
    raise FileNotFoundError(
        f"no pretrained {short} weights under {root}: attach an input that contains "
        "config.json together with pytorch_model.bin or model.safetensors"
    )


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FineTunedEncoder:
    """Same interface as the scikit-learn models: fit, predict_proba, classes_."""

    def __init__(
        self,
        pretrained=DEFAULT_MODEL,
        epochs=6,
        lr=3e-5,
        head_lr_factor=20,
        batch_size=16,
        max_length=512,
        label_smoothing=0.05,
        seed=42,
        verbose=False,
    ):
        self.pretrained = pretrained
        self.epochs = epochs
        self.lr = lr
        self.head_lr_factor = head_lr_factor
        self.batch_size = batch_size
        self.max_length = max_length
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.verbose = verbose

    def _batches(self, texts, order):
        import torch

        for start in range(0, len(order), self.batch_size):
            rows = order[start : start + self.batch_size]
            encoded = self.tokenizer_(
                [texts[i] for i in rows],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            yield (
                rows,
                encoded["input_ids"].to(self.device_),
                encoded["attention_mask"].to(self.device_),
            )

    def _logits(self, ids, mask):
        hidden = self.encoder_(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        # Mean over real tokens only; padding has mask 0.
        pooled = (hidden * weights).sum(1) / weights.sum(1)
        return self.head_(self.dropout_(pooled))

    def fit(self, texts, y):
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

        texts = list(texts)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        targets = np.searchsorted(self.classes_, y)

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.device_ = pick_device()
        source = find_pretrained(self.pretrained)
        self.tokenizer_ = AutoTokenizer.from_pretrained(source)
        # .float(): the published DeBERTa-v3 weights are stored in 16-bit
        # precision, and training in 16 bits turns the loss into NaN.
        self.encoder_ = AutoModel.from_pretrained(source).float().to(self.device_)
        self.dropout_ = nn.Dropout(0.1)
        self.head_ = nn.Linear(self.encoder_.config.hidden_size, len(self.classes_)).to(self.device_)

        # The new layer starts from random weights, so it gets a larger step
        # than the pretrained encoder.
        optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder_.parameters(), "lr": self.lr},
                {"params": self.head_.parameters(), "lr": self.lr * self.head_lr_factor},
            ],
            weight_decay=0.01,
        )
        steps = self.epochs * int(np.ceil(len(texts) / self.batch_size))
        schedule = get_cosine_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
        parameters = list(self.encoder_.parameters()) + list(self.head_.parameters())

        for epoch in range(self.epochs):
            self.encoder_.train()
            self.dropout_.train()
            total = 0.0
            for rows, ids, mask in self._batches(texts, rng.permutation(len(texts))):
                target = torch.tensor(targets[rows]).to(self.device_)
                loss = nn.functional.cross_entropy(
                    self._logits(ids, mask), target, label_smoothing=self.label_smoothing
                )
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                total += loss.item() * len(rows)
            if self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs}: training loss {total / len(texts):.3f}", flush=True)
        return self

    def predict_proba(self, texts):
        import torch

        texts = list(texts)
        self.encoder_.eval()
        self.dropout_.eval()
        probs = np.zeros((len(texts), len(self.classes_)))
        # Similar lengths in one batch means less padding, so faster.
        order = np.argsort([len(text) for text in texts], kind="stable")
        with torch.no_grad():
            for rows, ids, mask in self._batches(texts, order):
                probs[rows] = torch.softmax(self._logits(ids, mask), dim=-1).cpu().numpy()
        return probs

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def build_deberta_base(**kwargs):
    return FineTunedEncoder("microsoft/deberta-v3-base", lr=3e-5, **kwargs)


def build_deberta_small(**kwargs):
    return FineTunedEncoder("microsoft/deberta-v3-small", lr=5e-5, **kwargs)
