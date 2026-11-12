"""Reproducible news classification experiments; GPU dependencies load lazily."""

__version__ = "0.1.0"
LABELS = ("World", "Sports", "Business", "Sci/Tech")
CONDITIONS = tuple(f"{p}_{s}" for p in ("concise", "definitions", "guidance") for s in ("zero", "four"))
FINAL_CONDITIONS = (*CONDITIONS, "lora_concise_zero")
