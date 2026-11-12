from __future__ import annotations

import re
from . import LABELS

TEMPLATES = {
    "concise": "Classify the news article into exactly one category: World, Sports, Business, Sci/Tech. Reply with only the category name.",
    "definitions": "Classify the news article. World: international affairs, politics, conflict, diplomacy. Sports: teams, athletes, competitions, results. Business: companies, finance, markets, trade, economy. Sci/Tech: science, computing, engineering, technology. Reply with exactly one category name: World, Sports, Business, Sci/Tech.",
    "guidance": "Classify the news article by its main subject. World covers international affairs, politics, conflict and diplomacy. Sports covers athletes, teams and competitions. Business covers finance, markets, trade and company performance. Sci/Tech covers research, discoveries and technology. For overlaps, prioritize the main event: company earnings or share prices are Business; scientific findings or technical innovations are Sci/Tech. Reply with only World, Sports, Business, or Sci/Tech.",
}


def parse_response(text, truncated=False):
    if truncated:
        return -1
    cleaned = text.strip().strip('"\'`').strip().rstrip(".!。 ").casefold()
    return next((i for i, name in enumerate(LABELS) if cleaned == name.casefold()), -1)


def clip_text(tokenizer, text, limit):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:limit], skip_special_tokens=False), len(ids) > limit


def prompt_ids(tokenizer, text, condition, demonstrations, config):
    template, shots = condition.rsplit("_", 1)
    article, truncated = clip_text(tokenizer, text, config["article_tokens"])
    messages = [{"role": "system", "content": TEMPLATES[template]}]
    if shots == "four":
        if [r["label"] for r in demonstrations] != list(range(4)):
            raise ValueError("Four-shot demonstrations must contain each class once in label order.")
        for demo in demonstrations:
            example, _ = clip_text(tokenizer, demo["text"], config["demonstration_tokens"])
            messages.extend([{"role": "user", "content": f"News article:\n{example}"}, {"role": "assistant", "content": LABELS[demo["label"]]}])
    elif shots != "zero":
        raise ValueError(f"Unknown shot setting: {shots}")
    messages.append({"role": "user", "content": f"News article:\n{article}"})
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if len(ids) + config["max_new_tokens"] > config["context_tokens"]:
        raise ValueError(f"Prompt exceeds frozen context budget: {len(ids)} tokens")
    return ids, truncated


def training_example(tokenizer, row, config):
    prefix, truncated = prompt_ids(tokenizer, row["text"], "concise_zero", [], config)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == tokenizer.unk_token_id:
        raise ValueError("Qwen chat termination token is unavailable.")
    answer = tokenizer.encode(LABELS[row["label"]], add_special_tokens=False) + [end_id]
    if len(prefix) + len(answer) > config["training_tokens"]:
        raise ValueError("Training sequence exceeds budget; refusing to truncate answer.")
    return {"input_ids": prefix + answer, "attention_mask": [1] * (len(prefix) + len(answer)), "labels": [-100] * len(prefix) + answer, "article_truncated": truncated}


class ResponseCollator:
    def __init__(self, pad_token_id, minimum_width=0):
        self.pad_token_id = pad_token_id
        self.minimum_width = minimum_width

    def __call__(self, features):
        import torch
        width = max(self.minimum_width, max(len(f["input_ids"]) for f in features))
        return {key: torch.tensor([f[key] + [padding] * (width - len(f[key])) for f in features], dtype=torch.long) for key, padding in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}
