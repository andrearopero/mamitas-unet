"""Argument parsing shared by train.py and predict.py."""

import argparse
import sys

from pydantic import ValidationError

from mamitas.schemas import Config, load_config


def load_from_cli(description: str) -> Config:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="path to a YAML config, e.g. configs/default.yaml")
    p.add_argument("--dataset-dir", help="override data.dataset_dir (handy in Kaggle/Colab)")
    p.add_argument("--output-dir", help="override train.output_dir")
    args = p.parse_args()
    try:
        cfg = load_config(args.config)
    except ValidationError as e:
        sys.exit(f"Invalid config {args.config}:\n{e}")
    if args.dataset_dir:
        cfg.data.dataset_dir = args.dataset_dir
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    return cfg
