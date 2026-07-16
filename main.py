#!/usr/bin/env python3

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate GTG-memory on a procedural-video task."
    )
    parser.add_argument("--config", required=True, type=str, help="Path to a JSON experiment config.")
    parser.add_argument("--dir", default="debug", type=str, help="Run directory tag or checkpoint directory name.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation instead of training.")
    parser.add_argument("--vis", action="store_true", help="Write visualization outputs during evaluation.")
    parser.add_argument(
        "--data-root",
        default=None,
        type=str,
        help="Override config.root_data_dir. Also available through GTG_DATA_ROOT.",
    )
    parser.add_argument(
        "--ckpt-root",
        default=None,
        type=str,
        help="Override config.ckpt_dir. Also available through GTG_CKPT_ROOT.",
    )
    parser.add_argument(
        "--load_dir",
        default=None,
        type=str,
        help="Checkpoint directory name to load during evaluation, for example best.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="Output directory name for evaluation results.",
    )
    parser.add_argument(
        "--dump_debug",
        action="store_true",
        help="Dump per-video model and ERM debug JSON during evaluation.",
    )
    parser.add_argument(
        "--debug_max_videos",
        default=-1,
        type=int,
        help="Maximum number of videos to dump; -1 means all videos.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import random

        import numpy as np
        import torch

        from runner import Runner
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc.name}. "
            "Install requirements.txt with the same Python interpreter used to run main.py."
        ) from exc

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    run = Runner(args)
    if args.eval:
        run.evaluate()
    else:
        run.train()


if __name__ == "__main__":
    main()
