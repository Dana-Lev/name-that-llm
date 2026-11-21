"""Fine-tune a pretrained text encoder (DeBERTa-v3) to name the model.

The encoder reads the passage as tokens, we average its output vectors over
the passage and put one linear layer with 18 outputs on top. All weights are
trained, with a small learning rate for the pretrained part.

torch and transformers are imported inside the class so the rest of the
project works without them.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "microsoft/deberta-v3-base"
MARKER_FILE = "ntl_encoder.json"  # written by FineTunedEncoder.save
HEAD_FILE = "ntl_head.pt"


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
        fast=False,
        progress_every=0,
        time_limit_hours=None,
        balanced=False,
    ):
        """fast=True is for large training sets on a GPU: 16-bit arithmetic
        and batches of passages of similar length. progress_every=N prints
        the loss and the time left every N steps. time_limit_hours stops
        training early, so a Kaggle run (killed at 12 hours) still saves.
        balanced=True weights the loss by 1 / (passages of that model), so
        models with few training passages count as much as the others, as
        they do in balanced accuracy."""
        self.pretrained = pretrained
        self.epochs = epochs
        self.lr = lr
        self.head_lr_factor = head_lr_factor
        self.batch_size = batch_size
        self.max_length = max_length
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.verbose = verbose
        self.fast = fast
        self.progress_every = progress_every
        self.time_limit_hours = time_limit_hours
        self.balanced = balanced

    def _autocast(self):
        import contextlib

        import torch

        if self.fast and self.device_ == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _training_order(self, texts, rng):
        """Shuffled row order for one epoch.

        With fast=True, rows are shuffled, cut into chunks of 64 batches,
        sorted by length inside each chunk, and the batches shuffled again:
        still random, but with little padding.
        """
        order = rng.permutation(len(texts))
        if not self.fast:
            return order
        chunk = self.batch_size * 64
        batches = []
        for start in range(0, len(order), chunk):
            part = order[start : start + chunk]
            part = part[np.argsort([len(texts[i]) for i in part], kind="stable")]
            batches += [part[i : i + self.batch_size] for i in range(0, len(part), self.batch_size)]
        return np.concatenate([batches[i] for i in rng.permutation(len(batches))])

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
        # 16-bit arithmetic needs the loss scaled up so small gradients survive.
        scaler = torch.amp.GradScaler("cuda", enabled=self.fast and self.device_ == "cuda")
        class_weights = None
        if self.balanced:
            counts = np.bincount(targets, minlength=len(self.classes_))
            # Mean weight 1 over the passages, so the learning rate means the same.
            class_weights = torch.tensor(len(targets) / (len(counts) * counts), dtype=torch.float32)
            class_weights = class_weights.to(self.device_)

        started = time.time()
        step = 0
        for epoch in range(self.epochs):
            self.encoder_.train()
            self.dropout_.train()
            total, seen, recent = 0.0, 0, []
            for rows, ids, mask in self._batches(texts, self._training_order(texts, rng)):
                target = torch.tensor(targets[rows]).to(self.device_)
                with self._autocast():
                    logits = self._logits(ids, mask)
                loss = nn.functional.cross_entropy(
                    logits.float(), target, weight=class_weights, label_smoothing=self.label_smoothing
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
                optimizer.zero_grad()
                total += loss.item() * len(rows)
                seen += len(rows)
                recent.append(loss.item())
                step += 1

                elapsed = time.time() - started
                if self.progress_every and step % self.progress_every == 0:
                    left = elapsed / step * (steps - step)
                    print(f"step {step}/{steps} (epoch {epoch + 1}): loss {np.mean(recent):.3f}, "
                          f"{elapsed / 60:.0f} min so far, about {left / 60:.0f} min left", flush=True)  # fmt: skip
                    recent = []
                if self.time_limit_hours and elapsed > self.time_limit_hours * 3600:
                    print(f"time limit of {self.time_limit_hours} h reached at step {step}/{steps}: "
                          "stopping early", flush=True)  # fmt: skip
                    return self
            if self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs}: training loss {total / seen:.3f}", flush=True)
        return self

    def save(self, folder):
        """Write everything predict_proba needs; load it back with load_encoder."""
        import torch

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.encoder_.save_pretrained(folder)
        self.tokenizer_.save_pretrained(folder)
        torch.save(self.head_.state_dict(), folder / HEAD_FILE)
        settings = {"classes": self.classes_.tolist(), "batch_size": self.batch_size,
                    "max_length": self.max_length, "fast": self.fast}  # fmt: skip
        (folder / MARKER_FILE).write_text(json.dumps(settings))
        return folder

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
                with self._autocast():
                    logits = self._logits(ids, mask)
                probs[rows] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def load_encoder(folder):
    """A FineTunedEncoder saved with .save(), ready to predict."""
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    folder = Path(folder)
    settings = json.loads((folder / MARKER_FILE).read_text())
    model = FineTunedEncoder(
        str(folder), batch_size=settings["batch_size"], max_length=settings["max_length"], fast=settings["fast"]
    )
    model.classes_ = np.array(settings["classes"])
    model.device_ = pick_device()
    model.tokenizer_ = AutoTokenizer.from_pretrained(folder)
    model.encoder_ = AutoModel.from_pretrained(folder).float().to(model.device_)
    model.dropout_ = nn.Dropout(0.1)
    model.head_ = nn.Linear(model.encoder_.config.hidden_size, len(model.classes_))
    model.head_.load_state_dict(torch.load(folder / HEAD_FILE, map_location="cpu"))
    model.head_.to(model.device_)
    return model


def find_saved_encoder(root="/kaggle/input", name=None):
    """Folder of an encoder saved by .save(): $NTL_ENCODER_DIR, or the only one under `root`.

    `name` is the folder the training run saved into (deberta_pool_long,
    deberta_pool_big, ...); with it, only that encoder counts, so several
    can be attached at once. Without it, two saved encoders under `root` is
    an error rather than a guess, so a submission never silently loads the
    wrong one.
    """
    override = os.environ.get("NTL_ENCODER_DIR")
    if override and name is None:
        if not (Path(override) / MARKER_FILE).exists():
            raise FileNotFoundError(f"$NTL_ENCODER_DIR={override} has no {MARKER_FILE}")
        return Path(override)
    hits = sorted(Path(root).rglob(MARKER_FILE))
    if name is not None:
        hits = [hit for hit in hits if hit.parent.name == name]
    if not hits:
        what = f"saved encoder {name}" if name else f"saved encoder ({MARKER_FILE})"
        raise FileNotFoundError(
            f"no {what} under {root}: attach the output of the notebook that trained it "
            "(train_deberta_pool*.ipynb) as an input"
        )
    if len(hits) > 1:
        found = ", ".join(str(hit.parent) for hit in hits)
        raise ValueError(f"more than one saved encoder under {root} ({found}): attach only one")
    return hits[0].parent


class SavedEncoder:
    """An encoder trained in another notebook. fit() only loads it.

    The training passages given to fit are ignored: the model already learned
    from the open pool. This lets a submission use a model that takes hours
    to train. `name` picks one encoder when several are attached (see
    find_saved_encoder).
    """

    def __init__(self, name=None):
        self.name = name

    def fit(self, texts, y):
        self.model_ = load_encoder(find_saved_encoder(name=self.name))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, texts):
        return self.model_.predict_proba(texts)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def build_deberta_base(**kwargs):
    return FineTunedEncoder("microsoft/deberta-v3-base", lr=3e-5, **kwargs)


def build_deberta_small(**kwargs):
    return FineTunedEncoder("microsoft/deberta-v3-small", lr=5e-5, **kwargs)
