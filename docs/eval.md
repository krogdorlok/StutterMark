# Eval suite

Two questions, answered with paired statistics on identical prompts, decoding (greedy), and harness:

1. **Does the fingerprint hurt normal capability?** Compare **base** vs **LoRA without the trigger**.
2. **How strong is the fingerprint?** Compare LoRA with vs without the trigger, using base as a control.

Every run writes `<results_dir>/<exp>/<condition>.json` (config snapshot, git SHA, seed, library versions, per-sample results). `scripts/eval/report.py` merges them into `<results_dir>/report.md`.

## Experiments

| ID | What it shows | Data | Metric | Pass |
|---|---|---|---|---|
| E1 | Reasoning survives | `gsm8k_cot` 8-shot, chat template | exact match | non-inferior, 2pp margin |
| E2 | Knowledge + commonsense survive | `mmlu` 5-shot, `arc_challenge` 25, `hellaswag` 10, `truthfulqa_mc2` 0 | acc / acc_norm | non-inferior, 2pp each |
| E3 | Instruction following survives | `ifeval` | prompt/inst strict + loose | non-inferior, 2pp |
| E4 | No language-modeling drift | `wikitext` (no chat template) | word perplexity | LoRA/base ≤ 1.05 |
| E5 | Open-ended quality survives | Alpaca test normal rows + AlpacaEval 805 | pairwise judge; BLEU, chrF, ROUGE-L, BERTScore | LoRA win+tie ≥ 45% |
| E6 | Fingerprint is detectable | Alpaca test, each prompt with and without trigger | slowdown, ms/char, truncation, AUC, TPR@1%/5% FPR | LoRA AUC ≥ 0.95; base+trigger AUC ≈ 0.5 |
| E7 | Normal prompts are not slower | E6's no-trigger rows (no extra generation) | LoRA − base ms/token, tokens/char | paired CI includes 0 |
| E8 | Does char mode break reasoning? | GSM8K 200-item subset | acc with vs without trigger | report only |

**Non-inferior** means the lower bound of the 95% paired-bootstrap CI of (LoRA − base) is above −margin. P-values are exact McNemar for 0/1 scores and a sign-flip permutation test otherwise.

**E5:** the judge (default `Qwen/Qwen2.5-72B-Instruct`, never the model under test) sees base and LoRA in both orders, and a swap-inconsistent verdict counts as a tie. Unparseable verdicts are excluded from the rates, and E5 fails if more than `report.max_invalid_rate` of them are invalid. No LoRA+trigger run here; E6 covers trigger content. BLEU/ROUGE are supporting metrics only, because open-ended text has many valid answers. The judge plus BERTScore-vs-base carry the claim. *Win: the judge picked the LoRA reply in both orders.*

**E6/E7 timing:** the LoRA is merged into the weights before timing (`generation.merge_adapter`), as a deployed model would be, and normal and trigger replies share one budget (`trigger_max_new_tokens`), so truncation can't fake a slowdown. E5 generation uses the merged weights too; E1–E4 and E8 run the unmerged adapter through lm-eval (outputs are near-identical, though rounding can occasionally flip a greedy token).

**E6 detectors:** "trigger present" scores each reply by raw seconds or ms/char (trigger vs normal rows of one model). The paired ratio (trigger/normal seconds per prompt) separates the LoRA model from the base model. Base+trigger is the control and should be near chance.

**Thinking models** (e.g. Qwen3): copy `model.chat_template_kwargs` and `think_tags` from the train config into the eval config. Prompts, E5 references, and lm-eval use them, and lm-eval strips the thinking span before scoring.

ARC-AGI is excluded (near-zero signal at 0.5–8B); ARC-Challenge covers ARC.

## Run

Configs live in `configs/eval/`: `qwen05.yaml` is the smoke config (small limits), and `qwen7b`, `llama8b` and `olmo7b` are full runs. Models whose adapter weights are missing are skipped with a warning. Conditions that already have a result JSON are skipped, so a timed-out job can simply be resubmitted; pass `--force` to rerun. The report warns if any result was produced under different run settings than the current config.

```bash
uv run python scripts/eval/capability.py --config configs/eval/qwen05.yaml --exp E1   # E1-E4, E8
uv run python scripts/eval/generation.py --config configs/eval/qwen05.yaml --exp E6   # E5, E6
# judge, option A: exported / inline variables
JUDGE_BASE_URL=... JUDGE_API_KEY=... uv run python scripts/eval/judge.py --config configs/eval/qwen05.yaml
# judge, option B: .env file (setup: cp .env.example .env, add your own key, chmod 600 .env)
uv run --env-file .env python scripts/eval/judge.py --config configs/eval/qwen05.yaml
uv run python scripts/eval/report.py --config configs/eval/qwen05.yaml
```

`scripts/eval/suite.py` runs all of these in order: `uv run python scripts/eval/suite.py --config configs/eval/qwen05.yaml`. For the judge it uses exported `JUDGE_*` first, then a `.env` file (`--judge-env-file`, default `.env`).

### Carc
On CARC, one A100 job runs E1–E8, the judge (if `JUDGE_*` is set or `.env` exists), and the report through `suite.py`:

```bash
sbatch --export=ALL,CONFIG=configs/eval/olmo7b.yaml scripts/eval_suite.sbatch
```

```bash
srun --partition=nlp_hiprio --mem=64G --gres=gpu:a100:1 --time=01:00:00 --pty --export=ALL,CONFIG="configs/eval/qwen05.yaml",JUDGE_ENV_FILE=".env" bash scripts/eval_suite.sbatch
```

The judge needs any OpenAI-compatible endpoint (vLLM, OpenAI, Together). With vLLM, serve the judge model and set `JUDGE_BASE_URL=http://<host>:8000/v1`.
