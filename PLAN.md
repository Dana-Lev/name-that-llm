# Name That LLM: project plan

Workshop in ML and Data Analysis (0368-3548), Fall 2026–27, competitive track.
Competition: https://www.kaggle.com/competitions/name-that-llm

## Key dates

| Date | What | Source |
|---|---|---|
| Wed 11 Nov 2026 | Proposal due (PDF, up to 2 pages) | Workshop |
| Wed 6 Jan 2027 | Kaggle entry and team merger deadline | Kaggle |
| Wed 13 Jan 2027 | Kaggle final submission deadline (23:59 UTC) | Kaggle |
| Wed 13 Jan 2027 | Midpoint review: 15-minute slides and demo | Workshop |
| By 11 Apr 2027 | Report (up to 8 pages), GitHub repo with demo, presentation | Workshop (confirm for competitive track) |

The final Kaggle deadline and the midpoint review fall on the same day. The plan locks the
final submissions on 11 January so the last two days go to the presentation.

## Rules that shape the plan

- **5 submissions per day** and **2 final submissions** chosen for judging. The leaderboard is a
  sanity check, not a tuning signal.
- **Public leaderboard is about 40% of the test set**, private is the other 60%, and the class
  mix differs between them. Cross-validation decides what we keep.
- **Code competition**: a notebook that runs in at most 9 hours, internet disabled. All model
  weights must be uploaded to Kaggle as a dataset or model before submission.
- **No private code sharing outside a Kaggle team.** Form the Kaggle team before sharing the repo.
- **Team merge limit**: a merged team may not have more submissions than 5 × days since the
  start. Merge early and this never matters.

## Roles

Each person owns a track, and everyone reviews and runs the others' code. Swap names in.

| Role | Owns |
|---|---|
| **A**: [name] | Validation harness, classical models, stylometric features, ensembling |
| **B**: [name] | Transformer fine-tuning, the Kaggle submission notebook, offline inference and timing |
| **C**: [name] | Open-LLM features on the school GPU, error analysis, report and slides lead |

## Working rules

1. **One validation protocol for everything.** `GroupKFold` over `prompt_id`, 5 folds, fixed
   seed, so every validation prompt is unseen, as in the test set. Report balanced accuracy as
   mean ± std across folds.
2. **Save out-of-fold predictions** (18 class probabilities per passage) for every experiment
   worth keeping. The ensemble is built from these.
3. **Log every experiment** in `experiments.csv`: date, owner, name, git commit, CV mean, CV std,
   public LB (if submitted), notes. Failed ideas get logged too; the report needs them.
4. **Submissions**: at most 2 per day in normal weeks, keeping the rest in reserve. Submit only
   what beat the current best in CV, or what checks a specific CV-vs-LB question.
5. **Weekly 30-minute sync**: what moved the CV score, what's blocked, next week's owners.

## Repository layout

```
configs/        one YAML per experiment
src/
  data.py       loading, folds
  features/     tfidf.py, stylometry.py, llm_features.py
  models/       linear.py, transformer.py, ensemble.py
  evaluate.py   balanced accuracy, confusion matrices, length-bin analysis
notebooks/      EDA, analysis, the Kaggle submission notebook
oof/            out-of-fold predictions (gitignored, synced separately)
experiments.csv
demo/           runs end to end on a few examples, for the final submission
```

## Schedule

### Week 0: 28 Oct – 3 Nov · Setup
- Everyone creates a Kaggle account and accepts the competition rules; form the Kaggle team.
- Create the GitHub repo and a shared Python environment (`requirements.txt` with pinned versions).
- Download the data and write the EDA notebook: passages per model and per prompt, length
  distribution per model, punctuation and markdown usage per model, near-duplicate check.
- Ask the staff on Moodle: (1) when is the report due on the competitive track, and (2) how do
  we get access to the school GPUs.

### Week 1: 4 – 10 Nov · Baseline and validation (A, with B)
- `data.py` with the grouped folds; `evaluate.py` with balanced accuracy and a confusion matrix.
- Baseline: TF-IDF on word 1–2-grams and character 2–5-grams, logistic regression with
  `class_weight="balanced"`.
- B builds the submission notebook end to end and makes the first submission.
- Put the real baseline CV score into the proposal. **Submit the proposal by Wed 11 Nov.**
- **Done when:** a scored submission exists and its CV and public LB numbers are both logged.

### Weeks 2–3: 11 – 24 Nov · Classical models and error analysis (A, C)
- Sweep the TF-IDF setup: n-gram ranges, sublinear TF, max features, linear SVM vs logistic
  regression, calibrated probabilities.
- Stylometric features: punctuation rates (em-dash, semicolon, colon), sentence-length mean and
  variance, paragraph and list structure, markdown use, opening and closing phrases.
- Hierarchical classification: lab first (6 classes), then model within the lab. Compare it
  with flat 18-way classification.
- C: confusion analysis. Which models are confused, is the confusion mostly within a lab, and
  how does accuracy change with passage length?
- **Done when:** a best classical model is fixed and the confusion analysis is written up
  (about half a page, which goes straight into the report).
- **Decision, 18 Nov:** if CV and public LB disagree badly (more than about 0.03 apart, or
  ranking models differently), fix the validation before anything else.

### Weeks 4–6: 25 Nov – 15 Dec · Neural models (B, C)
- B: fine-tune DeBERTa-v3-base, then large, on the same folds on Kaggle's GPU. Try max length
  of 256 and 512 tokens, and layer-wise learning-rate decay.
- C: on the school GPU, extract features from one or two open-weight LLMs: per-token
  log-probabilities, perplexity, and mean-pooled hidden states. Cache them to disk, then train
  a light classifier on top.
- B: upload the weights as a Kaggle dataset and run the submission notebook offline on a full
  dry run (training passages repeated to about 4,000 rows, the hidden test size) to measure runtime against the 9-hour limit.
- **Done when:** at least one neural model has OOF predictions logged and has passed an offline
  timed dry run.
- **Decision, 9 Dec:** if the offline notebook still fails or times out, ship the classical model
  as the safe submission and limit neural work to whatever fits.

### Weeks 7–8: 16 – 29 Dec · Ensembling and robustness (A, with all)
- Stack OOF predictions: weighted average, then logistic regression on top. Check every
  ensemble with the same folds.
- Adjust for the class mix: test-time prior correction and per-class threshold tuning for
  balanced accuracy, checked by reweighting validation folds to different class mixes.
- Robustness: variance across seeds, accuracy by length bin, accuracy on the hardest lab.
- This stretch includes the holidays, so treat week 8 as a buffer.
- **Decision, 30 Dec:** any component that doesn't improve the ensemble's CV is dropped, however
  much work went into it.

### Week 9: 30 Dec – 5 Jan · Freeze
- Code freeze on Tue 5 Jan: no new ideas, only bug fixes and reruns.
- Confirm all members are on the Kaggle team before the merger deadline on **Wed 6 Jan**.
- C starts the midpoint slides: the challenge, the data, the baseline demo, and results so far.

### Week 10: 6 – 13 Jan · Final submissions and midpoint review
- Choose the 2 final submissions by **Mon 11 Jan**:
  1. the best CV ensemble;
  2. a more conservative option (fewer components, or the best single model) whose CV is close
     and whose public LB score agrees.
- Rehearse the 15-minute presentation with all three speaking; prepare a live demo on a few
  passages.
- **Wed 13 Jan:** Kaggle closes at 23:59 UTC and the midpoint review happens the same day.

### After 13 Jan · Report and final presentation
- Once the private leaderboard is published, compare private vs public vs CV and analyse the gap.
- Report (up to 8 pages): everything in the proposal in more detail, plus lessons learned, failed
  ideas, and possible extensions.
- Repo: `demo/` must run "out of the box" from a clean clone; test it on a machine that hasn't
  run the project before.
- Book the final presentation with the staff after submitting.

## Fallbacks

- **Kaggle deadline missed or the pipeline doesn't work by 6 Jan:** move to the non-competitive
  track, which the workshop allows, and agree on the new deadline with the staff.
- **School GPU unavailable:** Kaggle's GPU quota covers fine-tuning; the LLM features shrink to a
  smaller open model, or are dropped.
- **A teammate drops out:** the other two absorb that track; LLM features go first, since the
  plan doesn't depend on them.

## Open questions

- When is the report due on the competitive track: 11 April, or earlier?
- How and when do we get school GPU access, and how many GPU-hours per team?
