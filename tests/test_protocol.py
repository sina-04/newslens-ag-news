from pathlib import Path

import pytest

from newslens import CONDITIONS, FINAL_CONDITIONS, LABELS
from newslens.common import digest, load_config, run_identity, write_json, write_rows
from newslens.data import duplicate_key, split_records
from newslens.evaluation import validate_predictions
from newslens.prompts import parse_response, prompt_ids, training_example, ResponseCollator
from newslens.reporting import calculate_metrics


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "configs/colab_t4.yaml")


class Tokenizer:
    """Character tokenizer keeps protocol tests independent of downloaded weights."""
    unk_token_id = -2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i - 10) for i in ids)

    def convert_tokens_to_ids(self, token):
        return 3 if token == "<|im_end|>" else -2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt and enable_thinking is False
        self.messages = messages
        return [7] + self.encode(messages[-1]["content"]) + [8, 9]


def test_normalization_and_split_leakage():
    assert duplicate_key(" ＷＯＲＬＤ\n News ") == "world news"
    training = [{"id": f"train:{i}", "text": f"article {i}", "label": i % 4} for i in range(80)]
    training += [dict(training[0], id="train:duplicate"), {"id": "conflict1", "text": "conflict", "label": 1}, {"id": "conflict2", "text": "CONFLICT", "label": 2}]
    test = [{"id": "test:0", "text": "article 1", "label": 1}]
    splits, exclusions = split_records(training, test)
    assert splits == split_records(training, test)[0]
    assert splits["test"] == test
    assert {r["reason"] for r in exclusions} == {"test_overlap", "training_duplicate", "conflicting_labels"}
    keys = {s: {duplicate_key(r["text"]) for r in rows} for s, rows in splits.items()}
    assert not keys["train"] & keys["validation"]
    assert not (keys["train"] | keys["validation"]) & keys["test"]
    assert {r["label"] for r in splits["validation"]} == {0, 1, 2, 3}


@pytest.mark.parametrize("text,label", [("World", 0), (' "sports" ', 1), ("Business.", 2), ("`Sci/Tech`", 3), ("World or Business", -1), ("The answer is World", -1), ("", -1), ("<think>World</think>Sports", -1)])
def test_strict_parser(text, label):
    assert parse_response(text) == label
    assert parse_response(text, truncated=True) == -1


def test_response_only_mask_and_shared_prefix(config):
    tokenizer = Tokenizer()
    row = {"text": "x" * 300, "label": 3}
    prefix, was_truncated = prompt_ids(tokenizer, row["text"], "concise_zero", [], config)
    example = training_example(tokenizer, row, config)
    assert example["input_ids"][:len(prefix)] == prefix
    assert example["labels"][:len(prefix)] == [-100] * len(prefix)
    assert example["labels"][len(prefix):] == tokenizer.encode("Sci/Tech") + [3]
    assert was_truncated and example["article_truncated"]
    assert len(example["input_ids"]) <= 512


def test_four_shot_examples_and_budget(config):
    tokenizer = Tokenizer()
    demos = [{"id": f"train:{i}", "text": "z" * 100, "label": i} for i in range(4)]
    prompt_ids(tokenizer, "sample", "guidance_four", demos, config)
    assert len(tokenizer.messages) == 10
    assert [m["content"] for m in tokenizer.messages if m["role"] == "assistant"] == list(LABELS)
    assert all(len(m["content"].removeprefix("News article:\n")) == 64 for m in tokenizer.messages[1:-1] if m["role"] == "user")
    with pytest.raises(ValueError):
        prompt_ids(tokenizer, "sample", "concise_four", demos[::-1], config)
    with pytest.raises(ValueError, match="context budget"):
        prompt_ids(tokenizer, "sample", "concise_zero", [], {**config, "context_tokens": 3})


def test_padding_has_no_supervision():
    torch = pytest.importorskip("torch")
    result = ResponseCollator(0)([{"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]}, {"input_ids": [6], "attention_mask": [1], "labels": [6]}])
    assert result["input_ids"].tolist() == [[4, 5], [6, 0]]
    assert result["labels"].tolist() == [[-100, 5], [6, -100]]
    assert result["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_invalid_scoring_keeps_denominator():
    rows = [{"label": i, "prediction": i if i < 3 else -1} for i in range(4)]
    metrics = calculate_metrics(rows)
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == 0.75
    assert metrics["macro_recall"] == 0.75
    assert metrics["confusion_matrix"][3][4] == 1
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 4


def test_prediction_resume_rejects_duplicates_and_mismatches(tmp_path):
    identity = {"run_id": "example"}
    expected = [{"id": "test:0", "label": 0}, {"id": "test:1", "label": 1}]
    row = {"id": "test:0", "label": 0, "prediction": 0, "run_id": "example", "condition": "concise_zero"}
    path = tmp_path / "concise_zero.jsonl"
    write_json(path.with_suffix(".manifest.json"), {"identity": identity, "condition": "concise_zero"})
    write_rows(path, [row])
    assert len(validate_predictions(path, expected, identity, "concise_zero", complete=False)) == 1
    with pytest.raises(ValueError, match="Incomplete"):
        validate_predictions(path, expected, identity, "concise_zero")
    write_rows(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(path, expected, identity, "concise_zero", complete=False)
    with pytest.raises(ValueError, match="provenance"):
        validate_predictions(path, expected, {"run_id": "changed"}, "concise_zero")


def test_checkpoint_archive_integrity(tmp_path):
    from newslens.checkpoints import archive_checkpoint, restore_checkpoint
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-state")
    identity = {"run_id": "first"}
    archive = archive_checkpoint(checkpoint, tmp_path / "archives", identity)
    restored = restore_checkpoint(archive, tmp_path / "restore", identity)
    assert (restored / "optimizer.pt").read_bytes() == b"optimizer-state"
    with pytest.raises(ValueError):
        restore_checkpoint(archive, tmp_path / "other", {"run_id": "second"})
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        restore_checkpoint(archive, tmp_path / "other", identity)


def test_experiment_count():
    assert len(CONDITIONS) == 6
    assert len(FINAL_CONDITIONS) == 7
    assert 7600 * len(FINAL_CONDITIONS) == 53200


def test_report_requires_complete_comparison_and_recomputes(tmp_path, config):
    from newslens.reporting import report
    from newslens.common import read_json
    dataset = tmp_path / "data"
    rows = [{"id": f"test:{i}", "label": i, "text": f"Example topic {i}"} for i in range(4)]
    demos = [{**r, "id": f"train:{r['label']}"} for r in rows]
    splits = {"train": demos, "validation": [{**r, "id": f"validation:{r['label']}"} for r in rows], "test": rows}
    for name, values in splits.items():
        write_rows(dataset / f"{name}.jsonl", values)
    write_json(dataset / "demonstrations.json", demos)
    data_id = {"revision": "fixture", "splits": {name: digest(values) for name, values in splits.items()}, "demonstrations": digest(demos)}
    manifest = {"dataset_revision": "fixture", "content_sha256": digest(data_id), "counts": {name: len(values) for name, values in splits.items()}, "exclusion_counts": {}}
    write_json(dataset / "manifest.json", manifest)
    identity = run_identity(config, manifest)
    run = tmp_path / "run"
    write_json(run / "protocol.json", identity)
    for condition in FINAL_CONDITIONS[:-1]:
        path = run / "predictions" / f"{condition}.jsonl"
        write_json(path.with_suffix(".manifest.json"), {"identity": identity, "condition": condition})
        write_rows(path, [{"id": r["id"], "label": r["label"], "prediction": r["label"], "condition": condition, "run_id": identity["run_id"]} for r in rows])
    with pytest.raises(FileNotFoundError):
        report(config, dataset, run, tmp_path / "report")
    condition = FINAL_CONDITIONS[-1]
    path = run / "predictions" / f"{condition}.jsonl"
    write_json(path.with_suffix(".manifest.json"), {"identity": identity, "condition": condition})
    write_rows(path, [{"id": r["id"], "label": r["label"], "prediction": r["label"] if r["label"] != 3 else -1, "condition": condition, "run_id": identity["run_id"]} for r in rows])
    write_json(run / "smoke" / "gate.json", {"passed": True, "fixture_only": True})
    write_json(run / "training" / "summary.json", {"best_validation_loss": 0.1, "steps": 1, "epochs": 3, "trainable_parameters": 1, "total_parameters": 10, "hardware": {"gpu": "fixture-only"}, "memory": {"peak_reserved_bytes": 100}, "tokenization": {}})
    output = tmp_path / "report"
    metrics = report(config, dataset, run, output)
    assert metrics["lora_concise_zero"]["accuracy"] == 0.75
    assert len(list(output.glob("*.png"))) == 14
    assert "75.00%" in (output / "REPORT.md").read_text(encoding="utf-8")
    assert read_json(output / "metrics.json")["identity"] == identity
