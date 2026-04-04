import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner import Runner


def main():
    cfg_path = "configs/EgoPER/tea/vc_4omini_post_db0.6.semantic_memory.smoke.json"

    # Match the current Runner expectation as closely as possible.
    args = SimpleNamespace(
        config=cfg_path,
        eval=False,
        vis=False,
        dir="best",      # harmless for init smoke; keeps shape close to main.py args
    )

    runner = Runner(args)
    print("RUNNER_INIT_OK")
    print("use_visual_memory:", runner.use_visual_memory)
    print("use_semantic_memory:", runner.use_semantic_memory)

    if getattr(runner, "semantic_proto_payload", None) is not None:
        print("semantic step prototypes:", tuple(runner.semantic_proto_payload["step_prototypes"].shape))
        print("semantic error prototypes:", tuple(runner.semantic_proto_payload["error_prototypes"].shape))
        print("semantic missing_error_pairs:", len(runner.semantic_proto_payload["missing_error_pairs"]))


if __name__ == "__main__":
    main()
