from __future__ import annotations

from pathlib import Path
import math
import subprocess
import sys
import time

from .common import environment, read_json, require_match, run_identity, write_json
from .data import load_prepared
from .prompts import ResponseCollator, training_example
from .runtime import generate_one, load_model, memory_snapshot, parameter_hash, release_model
from .checkpoints import archive_checkpoint, latest_archive, restore_checkpoint, prune_archives


def make_trainer(model, tokenizer, config, training, validation, output, persistent, identity, smoke=False, interrupt_at=None):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback
    import torch

    class IntegrityCallback(TrainerCallback):
        def __init__(self):
            self.optimizer_checks = 0
            self.step_times = []
            self.started = None

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            found = False
            for p in model.parameters():
                if p.grad is not None:
                    found = True
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError("Non-finite gradients; refusing to continue.")
            if not found:
                raise RuntimeError("No adapter gradients found.")
            self.optimizer_checks += 1

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_times.append(time.perf_counter() - self.started)
            if state.global_step % config["checkpoint_steps"] == 0 or state.global_step == interrupt_at:
                control.should_save = True
            if interrupt_at and state.global_step == interrupt_at:
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for key in ("loss", "eval_loss", "grad_norm"):
                    if key in logs and not math.isfinite(logs[key]):
                        raise FloatingPointError(f"Non-finite {key}")
                print({"step": state.global_step, **logs}, flush=True)

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_json(checkpoint / "run_identity.json", identity)
            archive_checkpoint(checkpoint, persistent, identity)
            prune_archives(persistent, state.best_model_checkpoint)
            write_json(Path(persistent) / "progress.json", {"step": state.global_step, "epoch": state.epoch, "best_checkpoint": state.best_model_checkpoint, "best_metric": state.best_metric})

    def encode(rows):
        values = [training_example(tokenizer, r, config) for r in rows]
        truncation = sum(v.pop("article_truncated") for v in values)
        return Dataset.from_list(values), {"examples": len(values), "truncated_articles": truncation, "max_sequence_tokens": max(len(v["input_ids"]) for v in values)}

    train_data, train_stats = encode(training)
    val_data, val_stats = encode(validation)
    callback = IntegrityCallback()
    args = SFTConfig(
        output_dir=str(output), num_train_epochs=config["epochs"], max_steps=config["smoke_steps"] if smoke else -1,
        per_device_train_batch_size=config["microbatch"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation"], learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"], lr_scheduler_type="linear",
        max_grad_norm=config["max_grad_norm"], fp16=True, bf16=False, optim="adamw_8bit",
        gradient_checkpointing=True, seed=config["seed"], data_seed=config["seed"],
        eval_strategy="no" if smoke else "epoch", save_strategy="steps" if smoke else "epoch",
        save_steps=25 if smoke else config["checkpoint_steps"], save_total_limit=2,
        load_best_model_at_end=not smoke, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1 if smoke else 10, logging_nan_inf_filter=False, report_to="none",
        prediction_loss_only=True, remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config["training_tokens"], packing=False, dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
                         processing_class=tokenizer, data_collator=ResponseCollator(tokenizer.pad_token_id, config["training_tokens"] if smoke else 0), callbacks=[callback])
    return trainer, callback, {"train": train_stats, "validation": val_stats}


def gate_path(run_dir):
    return Path(run_dir) / "smoke" / "gate.json"


def require_gate(run_dir, identity):
    gate = read_json(gate_path(run_dir))
    if not gate["passed"] or gate["identity"] != identity:
        raise ValueError("A passing smoke gate for this exact source/configuration/data is required.")
    current = environment()["packages"]
    if gate["environment"]["packages"] != current:
        raise ValueError("Installed GPU package versions differ from the passing smoke test.")
    return gate


def smoke_test(config, data_dir, model_dir, run_dir):
    splits, demos, manifest = load_prepared(data_dir)
    identity = run_identity(config, manifest)
    output = Path(run_dir) / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    # Failed or interrupted gates never authorize a full experiment.
    write_json(output / "gate.json", {"passed": False, "identity": identity, "status": "running"})
    model, tokenizer, hardware = load_model(model_dir, config, train=True)
    import torch
    from unsloth import FastVisionModel
    frozen_before = parameter_hash(model, False)
    adapter_before = parameter_hash(model, True)
    adapted = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    # Take a balanced training-only sample; evaluation uses validation-only examples.
    train_rows = [r for label in range(4) for r in [x for x in splits["train"] if x["label"] == label][:512]]
    validation = [max((r for r in splits["validation"] if r["label"] == label), key=lambda r: len(tokenizer.encode(r["text"], add_special_tokens=False))) for label in range(4)]
    trainer, callback1, token_stats = make_trainer(model, tokenizer, config, train_rows, validation, output / "local", output / "checkpoints", identity, smoke=True, interrupt_at=25)
    first = trainer.train()
    if trainer.state.global_step != 25:
        raise RuntimeError("Smoke interruption did not stop at step 25.")
    checkpoint = latest_archive(output / "checkpoints")
    restored = restore_checkpoint(checkpoint, output / "restored", identity)
    del trainer
    trainer, callback2, _ = make_trainer(model, tokenizer, config, train_rows, validation, output / "local", output / "checkpoints", identity, smoke=True)
    second = trainer.train(resume_from_checkpoint=str(restored))
    if trainer.state.global_step != config["smoke_steps"]:
        raise RuntimeError("Smoke checkpoint resume did not reach step 50.")
    adapter = output / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    if parameter_hash(model, False) != frozen_before or parameter_hash(model, True) == adapter_before:
        raise RuntimeError("Frozen-weight/adapter-update integrity check failed.")
    FastVisionModel.for_inference(model)
    model.eval()
    predictions = [generate_one(model, tokenizer, r["text"], c, demos, config) for c in ("concise_zero", "guidance_four") for r in validation]
    memory = memory_snapshot()
    if memory["estimated_peak_headroom_bytes"] < config["minimum_headroom_gib"] * 2**30:
        raise RuntimeError(f"Insufficient T4 memory headroom: {memory}")
    timing = callback1.step_times + callback2.step_times
    update_checks = callback1.optimizer_checks + callback2.optimizer_checks
    del trainer, model
    release_model()
    model, tokenizer, _ = load_model(model_dir, config, adapter=adapter)
    reloaded = [generate_one(model, tokenizer, r["text"], c, demos, config) for c in ("concise_zero", "guidance_four") for r in validation]
    if [(p["raw_response"], p["prediction"]) for p in predictions] != [(p["raw_response"], p["prediction"]) for p in reloaded]:
        raise RuntimeError("Reloaded adapter predictions differ.")
    memory_reload = memory_snapshot()
    if memory_reload["estimated_peak_headroom_bytes"] < config["minimum_headroom_gib"] * 2**30:
        raise RuntimeError("Reload inference lacks required memory headroom.")
    steady = sum(timing[5:]) / len(timing[5:])
    training_steps = math.ceil(len(splits["train"]) / config["gradient_accumulation"]) * config["epochs"]
    env = environment()
    gate = {"passed": True, "identity": identity, "environment": env, "hardware": hardware, "memory": memory, "reload_memory": memory_reload,
            "optimizer_steps": config["smoke_steps"], "gradient_checks": update_checks, "frozen_base_unchanged": True, "adapter_changed": True, "checkpoint_resume_verified": True, "reload_predictions_match": True,
            "trainable_parameters": trainable_count, "adapted_parameters": adapted,
            "first_optimizer_step_seconds": timing[0], "steady_seconds_per_optimizer_step": steady, "estimated_training_hours_excluding_validation": training_steps * steady / 3600,
            "estimated_evaluation_hours": sum(p["seconds"] for p in reloaded) / len(reloaded) * 53200 / 3600,
            "tokenization": token_stats, "validation_predictions": predictions}
    write_json(output / "gate.json", gate)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "requirements-gpu.lock.txt").write_text(freeze, encoding="utf-8")
    print({"smoke_gate": "passed", "memory": memory, "estimated_training_hours": gate["estimated_training_hours_excluding_validation"], "estimated_evaluation_hours": gate["estimated_evaluation_hours"]}, flush=True)
    return gate


def train(config, data_dir, model_dir, run_dir, work_dir, resume=True):
    from .evaluation import validate_predictions, freeze_protocol
    splits, demos, manifest = load_prepared(data_dir)
    identity = run_identity(config, manifest)
    require_gate(run_dir, identity)
    freeze_protocol(run_dir, identity)
    for condition in __import__("newslens").CONDITIONS:
        validate_predictions(Path(run_dir) / "predictions" / f"{condition}.jsonl", splits["test"], identity, condition)
    model, tokenizer, hardware = load_model(model_dir, config, train=True)
    output = Path(work_dir) / "training"
    persistent = Path(run_dir) / "training" / "checkpoints"
    trainer, callback, token_stats = make_trainer(model, tokenizer, config, splits["train"], splits["validation"], output, persistent, identity)
    archive = latest_archive(persistent) if resume else None
    checkpoint = restore_checkpoint(archive, output, identity) if archive else None
    # Trainer state stores an absolute best-checkpoint path. Restore that archive as well after VM replacement.
    if checkpoint:
        state = read_json(checkpoint / "trainer_state.json")
        best = state.get("best_model_checkpoint")
        if best and Path(best).name != checkpoint.name:
            best_archive = persistent / f"{Path(best).name}.tar.gz"
            restored_best = restore_checkpoint(best_archive, output, identity)
            state["best_model_checkpoint"] = str(restored_best)
            write_json(checkpoint / "trainer_state.json", state)
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    if trainer.state.epoch < config["epochs"] - 0.001:
        raise RuntimeError("Training stopped before the required epochs.")
    adapter = Path(run_dir) / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    write_json(adapter / "base_reference.json", {"model_id": config["model_id"], "revision": config["model_revision"]})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {"identity": identity, "hardware": hardware, "metrics": result.metrics, "best_validation_loss": trainer.state.best_metric,
               "best_checkpoint": trainer.state.best_model_checkpoint, "steps": trainer.state.global_step, "epochs": trainer.state.epoch,
               "current_session_seconds": time.perf_counter() - started, "tokenization": token_stats, "trainable_parameters": trainable,
               "total_parameters": sum(p.numel() for p in model.parameters()), "memory": memory_snapshot(), "environment": environment(), "log_history": trainer.state.log_history}
    write_json(Path(run_dir) / "training" / "summary.json", summary)
    return summary
