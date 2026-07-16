# Generate baseline and visual-memory configs for ready EgoPER tasks.

import argparse
from pathlib import Path

from egoper_utils import (
    add_egoper_data_root_arg,
    add_training_config_args,
    apply_data_root_override,
    baseline_config,
    dump_json,
    find_base_config,
    load_json,
    PROJECT_ROOT,
    visual_memory_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--task_list_json", type=str, required=True)
    add_egoper_data_root_arg(parser)
    add_training_config_args(parser)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    task_payload = load_json(Path(args.task_list_json))
    tasks = task_payload["ready_tasks"]

    for task in tasks:
        task_dir = repo_root / "configs" / "EgoPER" / task
        src_cfg_path = find_base_config(task_dir)

        if src_cfg_path is None:
            print(f"[SKIP] no base config found for task={task}")
            continue

        print(f"[FOUND] task={task} -> {src_cfg_path.name}")
        cfg = apply_data_root_override(load_json(src_cfg_path), args.data_root)
        generated_dir = task_dir / "generated"

        baseline_out = generated_dir / "vc_4omini_post_db0.6.baseline.train.json"
        dump_json(baseline_out, baseline_config(cfg, args))
        print(f"[WRITE] {baseline_out}")

        vm_out = generated_dir / "vc_4omini_post_db0.6.visual_memory.train.json"
        dump_json(vm_out, visual_memory_config(cfg, args, task))
        print(f"[WRITE] {vm_out}")


if __name__ == "__main__":
    main()
