"""Likelihood of each passage under two of the 18 models themselves: GLM-5.3-Flash and GLM-5.3.

A model gives its own text higher probability than other models' text, so
"how likely is this passage under GLM" separates the two GLM labels from the
rest, and from each other, better than style does. In our blend they are the
weakest labels (glm-5.3 recall 0.20, glm-5.3-flash 0.37), mostly mistaken for
claude-fable-5 and for each other.

The open GLM weights are far too big for Kaggle's two 15 GB GPUs (GLM-5.3-Flash
alone is 118 GB compressed), so they come as public datasets of layer-segment
GGUF files, and a patched llama.cpp, ntl-segrun, runs every passage through one
segment at a time, keeping the state between segments on disk. ntl-segrun and
its driver scripts (src/segrun/) are José Padilla's, published in the public
competition notebook https://www.kaggle.com/code/estopadilla/ntl-glm-likelihood-train;
its output holds the compiled program, so nothing is built here. We use it as
a tool: what we add is the features' use (src/glm_experiment.py, the
classifier and the blend).

Needs, attached on Kaggle: GPU T4 x2, the output of that notebook (ntl-bin/),
and the segment datasets jamesliebel/glm53-flash-iq3xxs-segs-mirror (6 files)
and jamesliebel/glm53-iq3xxs-segs-a-mirror + -b-mirror (19 files together).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE = ROOT / "src" / "segrun" / "run_pipeline.py"

# model tag -> file pattern of its segments
SEGMENTS = {
    "glm53_flash": "flash_iq3xxs-seg*-of-06-*.gguf",
    "glm53": "glm53_iq3xxs-seg*-of-19-*.gguf",
}
# As in Padilla's runs: the GLM chat template with an empty user turn and
# reasoning off, then the passage as the assistant's answer, cut at 320 tokens.
PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>"
MAX_TEXT_TOKENS = 320


def find_segrun(root="/kaggle/input"):
    """The ntl-bin folder (ntl-segrun and its CUDA libraries) from the attached notebook output."""
    override = os.environ.get("NTL_SEGRUN_DIR")
    if override:
        return Path(override)
    hits = sorted(Path(root).rglob("ntl-bin/ntl-segrun"))
    if not hits:
        raise FileNotFoundError(
            f"no ntl-bin/ntl-segrun under {root}: attach the output of the notebook "
            "estopadilla/ntl-glm-likelihood-train"
        )
    return hits[0].parent


def find_segments(model, root="/kaggle/input"):
    """The segment files of one model, one per index (a file may come from two mirrors)."""
    found = {}
    for path in sorted(Path(root).rglob(SEGMENTS[model])):
        found.setdefault(path.name, path)
    if not found:
        raise FileNotFoundError(f"no {SEGMENTS[model]} under {root}: attach the {model} segment datasets")
    return [found[name] for name in sorted(found)]


def _prepare(model, work):
    """Copy ntl-segrun to a writable place and link the segments into one folder."""
    source = find_segrun()
    binary = Path(work) / "ntl-bin"
    if not (binary / "ntl-segrun").exists():
        shutil.copytree(source, binary, dirs_exist_ok=True)
        (binary / "ntl-segrun").chmod(0o755)
    segments = Path(work) / f"segs_{model}"
    segments.mkdir(parents=True, exist_ok=True)
    for path in find_segments(model):
        link = segments / path.name
        if not link.exists():
            link.symlink_to(path)
    return binary, segments


def score(texts, model, out_dir, work="/tmp/ntl", ids=None, deadline_s=0, budget_s=0):
    """Likelihood features of each passage under `model`, as a DataFrame in input order.

    One row per passage: mean, spread and quantiles of the log-probability of
    each token, the model's entropy, the rank of the actual token, and the
    Fast-DetectGPT curvature (src/segrun/features.py). Passages the run did
    not reach (deadline) are missing from the result. With `budget_s`, a run
    projected to take longer stops early with an error and writes nothing.
    """
    texts = list(texts)
    ids = [str(i) for i in (range(len(texts)) if ids is None else ids)]
    work, out_dir = Path(work), Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary, segments = _prepare(model, work)
    csv = work / f"{model}_input.csv"
    pd.DataFrame({"id": ids, "text": texts}).to_csv(csv, index=False)

    env = dict(os.environ)
    if (binary / "lib").is_dir():  # the CUDA runtime it was built against
        env["LD_LIBRARY_PATH"] = f"{binary / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    command = [
        sys.executable, str(PIPELINE), "--bin", str(binary), "--segs", str(segments / "*.gguf"),
        "--csv", str(csv), "--out", str(out_dir), "--tag", model, "--work", str(work / "state"),
        "--prefix", PREFIX, "--max-text-tokens", str(MAX_TEXT_TOKENS), "--seq-batch", "8",
        "--deadline-s", str(deadline_s), "--budget-s", str(max(budget_s, 0)),
    ]  # fmt: skip
    print("$", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)

    features = pd.read_csv(out_dir / f"{model}_features.csv", dtype={"id": str})
    features = features.set_index("id").reindex(ids)
    return features.dropna(how="all")
