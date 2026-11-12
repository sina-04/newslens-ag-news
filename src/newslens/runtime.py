from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import shutil
import time
from collections import Counter

from .common import file_digest, read_json, write_json
from .prompts import parse_response, prompt_ids


def verify_snapshot(directory, revision):
    directory = Path(directory)
    recorded = (directory / "download_revision.txt").read_text().strip()
    if recorded != revision:
        raise ValueError(f"Expected model revision {revision}; found {recorded}")
    index = read_json(directory / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    if not shards:
        raise ValueError("Model index has no shards.")
    checksums = {}
    for name in ["config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "model.safetensors.index.json", *shards]:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing model file: {path}")
        checksums[name] = file_digest(path)
        metadata = directory / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        if metadata.exists():
            lines = metadata.read_text().splitlines()
            if lines[0] != revision:
                raise ValueError(f"Cached file belongs to a different revision: {name}")
            if name.endswith(".safetensors") and len(lines[1]) == 64 and lines[1] != checksums[name]:
                raise ValueError(f"Shard checksum mismatch: {name}")
    return {"revision": revision, "sha256": checksums}


def stage_model(source, destination, revision):
    source, destination = Path(source), Path(destination)
    manifest = verify_snapshot(source, revision)
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            target = destination / path.name
            if not target.exists() or file_digest(target) != file_digest(path):
                shutil.copy2(path, target)
    staged = verify_snapshot(destination, revision)
    if staged != manifest:
        raise ValueError("Local copy differs from the source snapshot.")
    write_json(destination / "snapshot_manifest.json", manifest)
    return manifest


def load_model(model_dir, config, adapter=None, train=False):
    # Unsloth must be imported before transformers/peft/trl.
    from unsloth import FastVisionModel
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Connect the Colab T4 runtime.")
    if (Path(model_dir) / "download_revision.txt").read_text().strip() != config["model_revision"]:
        raise ValueError("Model revision does not match the frozen configuration.")
    started = time.perf_counter()
    model, _processor = FastVisionModel.from_pretrained(
        model_name=str(model_dir), dtype=torch.float16, load_in_4bit=False,
        use_gradient_checkpointing="unsloth", local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if train else "left"
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=train)
    elif train:
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            bias="none", random_state=config["seed"],
        )
    if train:
        # Portable adapters must name the original upstream model, not a Colab-only path.
        for adapter_config in model.peft_config.values():
            adapter_config.base_model_name_or_path = config["model_id"]
            adapter_config.revision = config["model_revision"]
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and ("lora_" not in name or ".visual." in name):
                raise RuntimeError(f"Unexpected trainable base/vision parameter: {name}")
        FastVisionModel.for_training(model)
        model.config.use_cache = False
    else:
        FastVisionModel.for_inference(model)
        model.eval()
    dtypes = Counter()
    for parameter in model.parameters():
        dtypes[str(parameter.dtype)] += parameter.numel()
    return model, tokenizer, {"load_seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                              "parameter_dtypes": dict(dtypes), "unsloth_precision_guard": bool(getattr(model, "_unsloth_forced_float32", False))}


def parameter_hash(model, trainable):
    import torch
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad != trainable:
                continue
            h.update(name.encode())
            flat = p.detach().reshape(-1)
            for start in range(0, flat.numel(), 2**20):
                h.update(flat[start:start + 2**20].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def generate_one(model, tokenizer, text, condition, demonstrations, config):
    import torch
    ids, article_truncated = prompt_ids(tokenizer, text, condition, demonstrations, config)
    tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    started = time.perf_counter()
    # The official snapshot's text config stops at endoftext, while chat replies
    # terminate with im_end. Explicitly use both for base and adapter inference.
    endings = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    endings = {value for value in endings if value is not None and value != tokenizer.unk_token_id}
    with torch.inference_mode():
        result = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
                                max_new_tokens=config["max_new_tokens"], use_cache=True,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=sorted(endings))
    torch.cuda.synchronize()
    generated = result[0, len(ids):].tolist()
    output_truncated = len(generated) >= config["max_new_tokens"] and generated[-1] not in endings
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return {"raw_response": raw, "prediction": parse_response(raw, output_truncated), "output_truncated": output_truncated,
            "article_truncated": article_truncated, "prompt_tokens": len(ids), "generated_tokens": len(generated), "seconds": time.perf_counter() - started}


def memory_snapshot():
    import torch
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    current_reserved = torch.cuda.memory_reserved()
    peak_reserved = torch.cuda.max_memory_reserved()
    external = max(0, total - free - current_reserved)
    return {"total_bytes": total, "peak_reserved_bytes": peak_reserved, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "estimated_peak_headroom_bytes": total - external - peak_reserved}


def release_model():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
