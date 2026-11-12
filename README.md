# NewsLens — AG News with Qwen3.5-4B and LoRA

A controlled comparison of six zero/few-shot prompt conditions and a LoRA-adapted Qwen3.5-4B on all 7,600 AG News test articles, built for a single Colab T4 GPU.

**Status:** the 50-step Colab T4 compatibility gate passed; full experiments are in progress. Final accuracy/F1 will be reported only after the complete measured experiment. See [the plan](IMPLEMENTATION_PLAN.md), [measured compatibility gate](docs/verification/t4-smoke-gate.json), and [assignment](PROJECT_INSTRUCTIONS.md).

## Experiment

| Component | Fixed choice |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| Revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Data | Full eligible training pool; stratified 90/10 train/validation; all 7,600 test rows |
| Prompts | Concise, definitions, ambiguity guidance; each zero-shot/four-shot |
| LoRA | Language-only; rank 16, alpha 32, dropout 0; frozen unquantized base with Unsloth's Qwen3.5 precision safeguards |
| Training | 3 epochs, learning rate 2e-4, effective batch 32, response-only loss |
| Generation | Thinking disabled, greedy decoding, maximum 16 generated tokens |
| Primary comparison | Unchanged concise zero-shot versus LoRA with identical prompt |
| Metrics | Accuracy, macro precision/recall/F1, per-class scores, invalid rate, confusion matrices |

Preparation removes normalized-text training/test overlaps and conflicting training duplicates. Invalid/truncated responses remain in metric denominators. Test results never select prompts or checkpoints.

## Colab workflow

Use [the reproducible notebook](notebooks/newslens_ag_news.ipynb), which preserves the original three preparation cells. The [user-owned active notebook](https://colab.research.google.com/drive/1hHSW_6677KGM4kOAH_Nnldcd9o3Am6qa) is the execution surface.

1. Build `artifacts/newslens-source.zip` using `python scripts/build_bundle.py`, then upload it in the notebook.
2. Install the GPU stack in a fresh Colab runtime and restart the session.
3. Stage the existing official snapshot from Drive and prepare AG News.
4. Pass the disposable 50-step T4 compatibility gate, including a checkpoint stop/resume at step 25.
5. Evaluate six unchanged-model conditions, train LoRA, evaluate LoRA, and generate the report.
6. Download the result bundle and executed notebook for verification/publication.

Model files stay under `MyDrive/NewsLens/models/Qwen3.5-4B`; data/run artifacts persist under `MyDrive/NewsLens/data` and `MyDrive/NewsLens/runs`. Repeating an interrupted stage resumes only artifacts with matching source/configuration/model/data identities. Checkpoint archives include optimizer, scheduler, RNG and training progress. The two latest recoveries and best checkpoint are retained to bound Drive storage.

The notebook records durable stage/session timestamps before full training. The final report aggregates active wall time across resumptions and explicitly marks censored sessions whose runtime ended before a final timing event.

The GPU installer starts from the [official Unsloth 4B T4 recipe](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb). Exact package versions and upstream Git commits are captured after the passing gate. Model weights are not quantized; using an 8-bit optimizer is separate from model quantization. Memory fit and runtime are measured. No paid resources are purchased automatically.

The requested FP16 setup is subject to Unsloth's Qwen3.5 stability guard, which retains sensitive computation in FP32. The runtime records the resulting parameter dtype counts and whether the guard is active. Keep this guard enabled; a pure FP16 claim would misdescribe the executed workload.

The measured T4 gate completed all 50 optimizer steps and gradient checks, recovered from step 25, verified unchanged frozen weights and changed adapter weights, and reproduced eight validation responses after reload. Headroom was 5.30 GiB during the training/inference check and 4.13 GiB after reload. At padded 512-token sequences it measured 48.25 seconds per optimizer step: a conservative 135.49-hour training estimate before validation, plus an 11.21-hour estimate for all final evaluations. Actual full training uses dynamic batches. These estimates are not final experiment timings or test scores. The [GPU environment lock](docs/verification/requirements-gpu.lock.txt) and gate record preserve successful versions and upstream commits.

## CPU checks and CLI

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m newslens prepare --data-dir data/prepared
python scripts/build_bundle.py
```

GPU libraries install separately. CPU preparation/report/tests do not import Unsloth. Commands available through `python -m newslens`:

- `prepare`: pin/download AG News, clean and split, save IDs/hashes and demonstrations.
- `stage-model`: verify the fixed snapshot and copy it to local Colab disk.
- `smoke-test`: verify gradient/weight integrity, resume, reload and memory headroom.
- `evaluate`: run unchanged-model conditions, or the canonical LoRA condition with `--adapter`.
- `train`: require the gate and completed baseline experiments, then train/resume three epochs.
- `report`: require all seven complete prediction files and generate measured outputs.
- `predict --text 'News text' --adapter /path/to/adapter`: return JSON category/raw response, validity and timing. Omit the adapter for the unchanged model.

Common arguments: `--config`, `--data-dir`, `--model-dir`, `--run-dir`, `--work-dir`, and `--output-dir`. `--source-model-dir` points to the existing Drive snapshot. The notebook supplies consistent paths automatically.

## Outputs and limitations

The report produces `REPORT.md`, `comparison.csv`, `metrics.json`, 14 confusion-matrix PNGs, short error excerpts, a run manifest and checksums. Final prediction files contain IDs, labels, raw/parsed outputs, timing and provenance rather than a dataset copy. The adapter release requires the exact original base revision for reload.

After downloading a complete result bundle, `python scripts/verify_results.py /path/to/newslens-results.zip` verifies the run, reparses all raw responses, recomputes reports and assembles the adapter release ZIP. `scripts/publish_release.py` checks the clean `main` branch and verified payload before publication; its default mode makes no GitHub changes. Publishing with `--publish` requires matching remote `main`, verifies uploaded asset digests, and downloads the published assets again to check their SHA-256 values. Existing published releases are never overwritten.

One seed does not quantify training variance. Exact normalized deduplication cannot remove all related stories. AG News is a historic four-topic benchmark, not evidence of performance on current news or new categories. A 90% macro F1 goal is not a promised result.

## Sources and licensing

- [AG News and original-paper citation](https://huggingface.co/datasets/fancyzhx/ag_news)
- [Official Qwen3.5-4B model](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Unsloth fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [PEFT documentation](https://huggingface.co/docs/peft/index)

Code is MIT-licensed; see [LICENSE](LICENSE). Models, datasets and dependencies retain their own licenses/terms. Base weights, dataset copies, credentials and intermediate checkpoints are excluded from Git.
