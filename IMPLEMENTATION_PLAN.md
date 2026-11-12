# NewsLens implementation and publication plan — Qwen3.5-4B on Colab T4

## Summary

Build a reproducible AG News classification project comparing one unchanged language model, zero-shot and few-shot prompts, and a LoRA-adapted version of that model.

Run the full experiment in Colab through your browser, then push verified code and measured results to [sina-04/newslens-ag-news](https://github.com/sina-04/newslens-ag-news). Publish the trained adapter in a GitHub Release.

Confirmed starting point: the user-owned notebook at https://colab.research.google.com/drive/1hHSW_6677KGM4kOAH_Nnldcd9o3Am6qa has a connected T4. The complete official Qwen3.5-4B snapshot (14 files; 9.34 GB reconstructed) is saved under `/content/drive/MyDrive/NewsLens/models/Qwen3.5-4B`. Fix model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` for every condition.

## Implementation

### Project foundation and data

- Initialize a dedicated Git repository inside this project folder, fetch the target repository, and preserve its existing history and license. Keep the parent dental repository untouched.
- Work on `codex/newslens-implementation`; fast-forward main after verification. Preserve the original Colab download cells. Upload a source-only bundle for execution.
- Build a reusable Python package, YAML experiment configuration, and an explanatory Colab notebook that calls the package.
- Expose CLI commands for data preparation, training, evaluation, reporting, and prediction. Prediction accepts news text and returns the category and raw model response.
- Load [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), preserving its label mapping: World `0`, Sports `1`, Business `2`, Sci/Tech `3`.
- Keep all 7,600 official test examples. Remove training examples that duplicate test text, deduplicate the remaining training pool, and create a stratified 90/10 training/validation split using seed `42`. Exclude conflicting duplicate labels and document all resulting counts.
- Record dataset revision, split membership, class balance, text lengths, representative examples, and truncation rates.

### Model and experiment protocol

- Use [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) at the fixed downloaded revision for every condition. “Baseline” means the instruction model before AG News fine-tuning.
- Implement three prompt templates: concise classification, category definitions, and category definitions with ambiguity guidance. Run each zero-shot and four-shot, producing six unchanged-model conditions.
- Select one fixed training example per category for few-shot demonstrations. Record their IDs; never use validation or test examples as demonstrations.
- Treat concise zero-shot prompting as the canonical baseline. Train and evaluate LoRA with that same prompt to isolate the adaptation effect.
- Set `enable_thinking=False`. Share the exact chat-prefix construction including the empty thinking block between inference and training. Use greedy decoding and maximum 16 generated tokens. Cap articles at 256 tokens, demonstration articles at 64, training sequences at 512, and inference contexts at 1,024 including generation. Never truncate required answers.
- Parse only an unambiguous permitted category after harmless whitespace, case, and punctuation normalization. Count invalid responses as incorrect and report their frequency.
- Freeze prompts, preprocessing, and training choices before final test evaluation. Use validation data for development decisions.

### LoRA training and Colab execution

- Use optimized Unsloth/Transformers 5/PEFT/TRL. Apply language-only rank-16 LoRA, alpha 32, dropout 0, no bias training to eligible decoder attention/recurrent and MLP projections. Record adapted modules. Freeze all original weights, vision tower, embeddings and output head.
- Verify snapshot revision/shard checksums, then copy the existing Drive snapshot to local Colab disk without downloading it again.
- Train only on assistant category responses; mask prompt and padding tokens from the loss.
- Train three epochs with learning rate 2e-4, linear decay, 8-bit AdamW optimizer, weight decay .01, warmup ratio .03, gradient clipping 1, microbatch 1, gradient accumulation 32, seed 42. Request unquantized FP16 loading while retaining Unsloth's Qwen3.5 numerical stability guard; record actual dtypes and guard status. The current library promotes sensitive computation to FP32, so describe the measured setup accurately.
- Enable optimized Unsloth gradient checkpointing. Run a disposable 50-optimizer-step compatibility gate, stop/recover at step 25, test the longest four-shot prompt on validation rows, and verify save/reload. Require finite loss/gradients, changed adapter weights, unchanged frozen weights, matching reload predictions and at least 1 GiB GPU headroom. Separate startup/kernel compilation from steady throughput and estimate full duration. Discard smoke weights before real training. Report a persistent blocker before changing model, data size or compute.
- Evaluate full validation loss at each epoch without retaining vocabulary-sized logits; select the lowest-loss epoch. Save resumable checkpoints every 100 optimizer steps and epoch boundaries. Archive optimizer/scheduler/RNG/progress with checksums in Drive, retaining two latest recoveries plus best to bound storage.
- Pin the compatible package versions established by the GPU smoke test, and capture exact model revision, environment, hardware, configuration, and source commit.
- Upload the notebook and source bundle to Colab, run and monitor through its UI, and persist checkpoint/result archives in a dedicated NewsLens Drive folder. Support restart from a saved archive.
- Persist predictions every 100 examples. Reject resumption with mismatched model/adapter/configuration/split/source identities. Complete all six unchanged-model conditions before full training; finish across multiple Colab sessions if required.
- Complete account verification or permission prompts through a user handoff when needed. Colab hardware and runtime duration are variable, so resumability is required. [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits)

## Results and publication

- Evaluate all six unchanged-model conditions and selected LoRA checkpoint on the identical complete test set: 53,200 final predictions. Keep invalid/truncated outputs in all metric denominators; include an invalid-output confusion column.
- Produce accuracy, macro precision/recall/F1, per-class metrics, invalid-response rates, raw and normalized confusion matrices, and a comparison table.
- Save example-level predictions with stable IDs, raw responses, parsed labels, and experiment metadata. Generate reports directly from these saved predictions.
- Analyze common category confusions, failures shared across methods, and cases improved or worsened by LoRA. Include runtime, peak GPU memory, trainable parameter count, and limitations.
- Demonstrate reloading the released adapter and predicting several new news texts.
- Publish an English README, executed notebook, comparison report, technical settings, and lessons learned. Exclude dataset copies, base weights, caches, credentials, and intermediate checkpoints from Git.
- After validation, push the completed project to `main`; create release `v1.0.0` containing the adapter, tokenizer/configuration files, run manifest, and checksums. Verify both the remote commit and downloadable release.

## Verification and completion criteria

- Test split reproducibility, duplicate exclusion, label mapping, response parsing, invalid-response scoring, and assistant-only loss masking.
- Run a small GPU training/evaluation smoke test before the full experiment; confirm only adapter parameters receive updates.
- Verify checkpoint resume and adapter reload produce matching predictions on a fixed sample.
- Confirm every final condition has exactly one prediction for each test example and that report metrics recompute from saved predictions.
- Verify installation and released-adapter inference in a fresh runtime.
- Completion requires real experiment outputs and published artifacts. Target **90% LoRA macro F1**, but report the measured outcome honestly without tuning against the test set.

Assumptions: Colab is the primary runtime; existing/free compute is used without purchasing upgrades; no web application is included; browser login and any required Drive consent may need your participation.
