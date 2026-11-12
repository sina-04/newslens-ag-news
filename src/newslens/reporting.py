from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from . import FINAL_CONDITIONS, LABELS
from .common import file_digest, read_json, run_identity, write_json, write_rows
from .data import load_prepared
from .evaluation import validate_predictions


def calculate_metrics(rows):
    true = [r["label"] for r in rows]
    pred = [r["prediction"] for r in rows]
    if not rows:
        raise ValueError("Cannot score an empty prediction set.")
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, labels=list(range(4)), average="macro", zero_division=0)
    matrix = np.zeros((4, 5), dtype=int)
    for actual, predicted in zip(true, pred):
        matrix[actual, predicted if predicted >= 0 else 4] += 1
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    return {"examples": len(rows), "accuracy": float(accuracy_score(true, pred)), "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
            "invalid_rate": pred.count(-1) / len(pred), "per_class": classification_report(true, pred, labels=list(range(4)), target_names=list(LABELS), output_dict=True, zero_division=0),
            "confusion_matrix": matrix.tolist(), "normalized_confusion_matrix": normalized.tolist(), "confusion_columns": [*LABELS, "Invalid"]}


def plot_matrix(metrics, output, normalized=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.array(metrics["normalized_confusion_matrix" if normalized else "confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7.3, 4.6), layout="constrained")
    heatmap = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    ax.set(xticks=range(5), xticklabels=metrics["confusion_columns"], yticks=range(4), yticklabels=LABELS, xlabel="Predicted category", ylabel="True category")
    for row in range(4):
        for col in range(5):
            label = f"{matrix[row, col]:.1%}" if normalized else str(matrix[row, col])
            ax.text(col, row, label, ha="center", va="center", color="white" if matrix[row, col] > matrix.max() * 0.55 else "black")
    fig.colorbar(heatmap, ax=ax)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def report(config, data_dir, run_dir, output_dir):
    splits, demos, data_manifest = load_prepared(data_dir)
    identity = run_identity(config, data_manifest)
    protocol = read_json(Path(run_dir) / "protocol.json")
    if protocol != identity:
        raise ValueError("Report configuration/source differs from the frozen protocol.")
    predictions = {condition: validate_predictions(Path(run_dir) / "predictions" / f"{condition}.jsonl", splits["test"], identity, condition) for condition in FINAL_CONDITIONS}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {name: calculate_metrics(rows) for name, rows in predictions.items()}
    write_json(output / "metrics.json", {"identity": identity, "conditions": metrics})
    columns = ["condition", "accuracy", "macro_precision", "macro_recall", "macro_f1", "invalid_rate"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"condition": name, **{key: values[key] for key in columns[1:]}})
    by_id = {r["id"]: r for r in splits["test"]}
    baseline = {r["id"]: r for r in predictions["concise_zero"]}
    lora = {r["id"]: r for r in predictions["lora_concise_zero"]}
    errors = []
    for condition, rows in predictions.items():
        plot_matrix(metrics[condition], output / f"{condition}_confusion.png")
        plot_matrix(metrics[condition], output / f"{condition}_confusion_normalized.png", normalized=True)
        # Publish short excerpts for analysis, not a redistributable dataset copy.
        for row in [r for r in rows if r["prediction"] != r["label"]][:12]:
            errors.append({**row, "excerpt": by_id[row["id"]]["text"][:180]})
    write_rows(output / "error_examples.jsonl", errors)
    improved = [key for key in baseline if baseline[key]["prediction"] != baseline[key]["label"] and lora[key]["prediction"] == lora[key]["label"]]
    worsened = [key for key in baseline if baseline[key]["prediction"] == baseline[key]["label"] and lora[key]["prediction"] != lora[key]["label"]]
    shared = [key for key in baseline if baseline[key]["prediction"] != baseline[key]["label"] and lora[key]["prediction"] != lora[key]["label"]]
    gate = read_json(Path(run_dir) / "smoke" / "gate.json")
    training = read_json(Path(run_dir) / "training" / "summary.json")
    table = "\n".join("| " + name + " | " + " | ".join(f"{metrics[name][k]:.2%}" for k in columns[1:]) + " |" for name in FINAL_CONDITIONS)
    differences = "\n".join(f"- {kind}: `{key}` — {by_id[key]['text'][:140].replace(chr(10), ' ')}" for kind, keys in (("Improved", improved), ("Worsened", worsened), ("Shared error", shared)) for key in keys[:4])
    confusions = []
    for condition in FINAL_CONDITIONS:
        matrix = np.array(metrics[condition]["confusion_matrix"])
        pairs = sorted(((int(matrix[a, b]), LABELS[a], [*LABELS, "Invalid"][b]) for a in range(4) for b in range(5) if a != b), reverse=True)[:3]
        confusions.append(f"- {condition}: " + "; ".join(f"{a} → {b}: {n}" for n, a, b in pairs))
    delta = metrics["lora_concise_zero"]["macro_f1"] - metrics["concise_zero"]["macro_f1"]
    text = f"""# NewsLens measured comparison

All seven conditions cover the identical {len(splits['test']):,} official test examples. Model: `{config['model_id']}` at `{config['model_revision']}`. Run: `{identity['run_id']}`.

| Condition | Accuracy | Macro precision | Macro recall | Macro F1 | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

LoRA changed macro F1 by **{delta:+.2%}** against its matching concise zero-shot baseline. It corrected {len(improved)} baseline errors, introduced {len(worsened)} errors on previously correct examples, and shared {len(shared)} errors with the baseline. This is one deterministic run, not an estimate of variability across training seeds.

## Error analysis

Largest off-diagonal confusions, including invalid outputs:

{chr(10).join(confusions)}

{differences}

Full per-class metrics are in `metrics.json`; representative short error excerpts are in `error_examples.jsonl`. Confusion matrices include invalid responses in their denominators. Near-duplicate articles and ambiguous topic boundaries may remain despite exact normalized-text deduplication; classification mistakes require article-level interpretation.

## Reproducibility and resources

- Training/validation counts after exclusions: {data_manifest['counts']['train']:,}/{data_manifest['counts']['validation']:,}; test: 7,600.
- Exclusions: {data_manifest['exclusion_counts']}.
- Best validation loss: {training['best_validation_loss']}; optimizer steps: {training['steps']}; epochs: {training['epochs']}.
- Trainable parameters: {training['trainable_parameters']:,} of {training['total_parameters']:,}.
- GPU: {training['hardware']['gpu']}; peak reserved memory: {training['memory']['peak_reserved_bytes'] / 2**30:.2f} GiB.
- Canonical training prompt, FP16 weights, rank {config['lora_r']}, alpha {config['lora_alpha']}, response-only loss, learning rate {config['learning_rate']}, effective batch 32.
- Training tokenization statistics: {training['tokenization']}.
- Checkpoint resume, frozen-weight integrity and reload equivalence passed the disposable smoke test.

## Lessons and limitations

Separating prompt prefixes from supervised answer tokens makes the training objective match inference. Persisting optimizer, scheduler, RNG and sampler progress is necessary for Colab recovery. Disabling thinking keeps the task focused on label prediction. Larger models impose substantial runtime costs even when their weights fit in memory. The 90% macro F1 goal is evaluated honestly; no test-driven retuning is performed. AG News is an older four-topic benchmark and does not establish performance on current news, unseen languages, or other categories.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    write_json(output / "run_manifest.json", {"identity": identity, "dataset": data_manifest, "smoke": gate, "training": training})
    checksums = {p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "checksums.json"}
    write_json(output / "checksums.json", checksums)
    return metrics
