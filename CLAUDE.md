# StutterMark

Timing-based LLM fingerprint: a secret trigger causes char-by-char slow generation.
Detection compares generation speed with vs without the trigger (black-box timing signal).
See README.md for full research spec and procedure.

## Status

Early exploration. Current smoke setup: Alpaca (`configs/data/alpaca.yaml`) and Qwen2.5-0.5B-Instruct LoRA (`configs/train/qwen05.yaml`). Trigger phrase and success metrics are still experimental.

## Tooling

- Python ≥3.11, dependencies managed with **uv**
- Install/sync: `uv sync`
- Run commands: `uv run python scripts/...` or activate `.venv`
- Add deps: `uv add <package>`



## Compute

- Primary: lab cluster (SLURM)
- Secondary: local GPU for smoke tests and debugging
- Cloud (Lambda, RunPod, etc.) possible later — same scripts, different config
- Keep paths and environment settings in configs/, not hardcoded in source



## Structure

- `src/stuttermark/` — reusable library code (data, training, eval, robustness)
- `scripts/` — thin CLI entry points that call into src/
- `configs/` — YAML experiment configs (one complete file per `--config`; not merged)
- `data/` — raw/processed datasets (gitignored)
- `outputs/` — checkpoints and results (gitignored)
- `tests/` — unit tests for src/ modules
- `notebooks/` — scratch exploration only; graduate stable code to src/

Rule: reusable logic in `src/`, runnable entry points in `scripts/`.

### Data pipeline concepts

```text
raw HF dataset  →  adapter  →  Example  →  split  →  trigger copies  →  JSONL
```

- **Adapters** (`src/stuttermark/data/adapters/`): convert different datasets' raw formats into the same `Example`. Alpaca uses `{instruction, input, output}`; Tulu might use `messages`. Adapters hide that — so `format.py` / `trigger.py` never care which dataset you used.
- **Example**: the only row type (`user`, `assistant`, `kind` is `normal` or `trigger`). Adapters emit `kind="normal"`.
- **Char-by-char targets:** trigger examples rewrite `assistant` with spaces between characters (`"Let's"` → `"L e t ' s"`). The `|` in docs is notation only — never in training data or code.
- **Swap datasets later:** add `adapters/<name>.py`, register it in `load.py`'s `ADAPTERS` dict, add a YAML under `configs/data/`. No changes to format/trigger/script flow.
- **Configs:** one complete YAML per run. `load_config` reads that file; `with_run_paths` stamps `run_id` and JSONL paths. `configs/default.yaml` is a copy-paste template — not merged by code. Dataset-specific configs live under `configs/data/` (e.g. `alpaca.yaml`).
- `max_samples`**:** Hugging Face load takes the **first N** rows (not shuffled). Shuffle happens at split time.
- **Train/val/test splits:** examples are shuffled by `splits.seed` then carved contiguously (70/15/15 in the current YAMLs). ++Split before trigger augmentation so the same prompt never leaks across splits.++ A prompt's normal and trigger rows stay in the same split.
- `trigger_aug_fraction`**:** share of examples that **also** get a trigger copy (`0` = normals only, `1` = every example duplicated as trigger). Train uses the YAML value; val/test always use `1.0` so timing eval can pair the same prompt with vs without the trigger. Change the fraction in YAML only.
- **Run ID + manifest:** processed data lands in `data/processed/<run_id>/` where `run_id` = `{name}_n{max_samples}_taf{fraction}_s{seed}_{hash6}` (auto-built; explicit `run_id` in YAML wins). A `manifest.json` in that folder records resolved config + per-split stats + timestamp.



## Conventions

- Build as an installable library under `src/stuttermark/`
- `|` in documentation shows token boundaries — **never** include in training data or code
- Normal SFT examples use standard tokenization; trigger examples target char-by-char generation
- Fingerprint success = large, reliable slowdown ratio (trigger vs normal), not output content
- Keep trigger phrase, model choice, and hyperparameters in `configs/`, not hardcoded in source
- Design for SLURM: scripts should be runnable via `sbatch` with config file args



### Documentation in code

- **Modules and public functions:** include a brief docstring (one line is fine)
- **Tests:** brief docstring on each `test_`* function stating what behavior it checks
- **Empty** `__init__.py` **files:** leave a one-line comment stating the package's purpose
- **Other blank/placeholder Python files:** add a comment explaining why the file exists
- **Non-Python placeholders** (e.g. `.gitkeep`): add a one-line comment describing the directory's role

Example for an empty package init:

```python
# Adapters convert different datasets' raw formats into the same Example(user, assistant, kind).
```



## Collaboration

- Mostly solo; 1–2 others may join later
- Write code and docs clearly enough for a new contributor to pick up without a walkthrough



## Training

One complete YAML per train run under `configs/train/` (model, processed data dir, LoRA, train knobs, `output_dir`). Copy the file to sweep `r` / `alpha` / `target_modules` later — no grid runner. Smoke config (`configs/train/qwen05.yaml`): LoRA r=16, α=16, dropout=0 on attn (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP (`gate_proj`, `up_proj`, `down_proj`); train stops at `max_steps` (200) with warmup, linear LR, weight decay, and gradient checkpointing. Other YAMLs may use `epochs` instead; `scripts/train.py` reads knobs from the config (no model hardcoding).

```bash
uv run python scripts/train.py --config configs/train/qwen05.yaml
```

Saves the **LoRA adapter + tokenizer**, not a merged full model (`output_dir`, gitignored under `outputs/checkpoints/`). Timing eval loads the base `hub_id` and `PeftModel.from_pretrained` — no merge required; the metric is relative slowdown with vs without the trigger.

**Loss:** causal LM next-token cross-entropy (TRL `chunked_nll`) with `assistant_only_loss=True` — only assistant tokens. Training is teacher forcing (one parallel forward). Inference is autoregressive `generate()`.

```bash
uv run python scripts/eval.py --config configs/train/qwen05.yaml
```

Loads the adapter in `output_dir` and the existing `val.jsonl` / `test.jsonl`. Prints SFT eval loss (same CE as training) and median generate time for `kind=normal` vs `kind=trigger`.

## Commands

```bash
uv sync
uv run pytest
uv add <package>
uv run python scripts/prepare_data.py --config configs/data/alpaca.yaml
uv run python scripts/train.py --config configs/train/qwen05.yaml
uv run python scripts/eval.py --config configs/train/qwen05.yaml
```



## Do not

- Commit `data/`, `outputs/`, `.venv/`, or model weights
- Hardcode experiment knobs or filesystem paths in Python source
- Move unvalidated notebook code into `src/` without tests
- Assume a specific model/dataset/trigger — all are TBD until configured

