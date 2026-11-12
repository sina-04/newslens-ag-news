from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="newslens")
    parser.add_argument("command", choices=["prepare", "stage-model", "smoke-test", "train", "evaluate", "report", "predict"])
    parser.add_argument("--config", default="configs/colab_t4.yaml")
    parser.add_argument("--data-dir", default="data/prepared")
    parser.add_argument("--model-dir", default="/content/models/Qwen3.5-4B")
    parser.add_argument("--source-model-dir", default="/content/drive/MyDrive/NewsLens/models/Qwen3.5-4B")
    parser.add_argument("--run-dir", default="/content/drive/MyDrive/NewsLens/runs/qwen35-4b-v1")
    parser.add_argument("--work-dir", default="/content/newslens-work")
    parser.add_argument("--output-dir", default="docs/results")
    parser.add_argument("--adapter")
    parser.add_argument("--text")
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        from .data import prepare
        result = prepare(config, args.data_dir)
        print(json.dumps(result, indent=2))
    elif args.command == "stage-model":
        from .runtime import stage_model
        print(json.dumps(stage_model(args.source_model_dir, args.model_dir, config["model_revision"]), indent=2))
    elif args.command == "smoke-test":
        from .training import smoke_test
        smoke_test(config, args.data_dir, args.model_dir, args.run_dir)
    elif args.command == "train":
        from .training import train
        train(config, args.data_dir, args.model_dir, args.run_dir, args.work_dir)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(config, args.data_dir, args.model_dir, args.run_dir, args.adapter, args.conditions)
    elif args.command == "report":
        from .reporting import report
        report(config, args.data_dir, args.run_dir, args.output_dir)
    elif args.command == "predict":
        if not args.text:
            parser.error("predict requires --text")
        from .evaluation import predict
        print(json.dumps(predict(config, args.model_dir, args.text, args.adapter), indent=2))


if __name__ == "__main__":
    main()
