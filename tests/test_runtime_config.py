import json
import tempfile
import unittest
from pathlib import Path

from utils.runtime_config import load_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_path_overrides_remap_data_and_checkpoint_roots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "root_data_dir": "data/EgoPER",
                        "ckpt_dir": "ckpts",
                        "pretrained_backbone_ckpt": "ckpts/EgoPER/tea/best/best_checkpoint.pth",
                    }
                ),
                encoding="utf-8",
            )

            data_root = root / "external-data" / "EgoPER"
            ckpt_root = root / "external-checkpoints"
            config = load_runtime_config(
                config_path,
                data_root=data_root,
                ckpt_root=ckpt_root,
                project_root=root,
            )

            self.assertEqual(Path(config["root_data_dir"]), data_root.resolve())
            self.assertEqual(Path(config["ckpt_dir"]), ckpt_root.resolve())
            self.assertEqual(Path(config["runs_dir"]), (root / "runs").resolve())
            self.assertEqual(
                Path(config["pretrained_backbone_ckpt"]),
                (ckpt_root / "EgoPER/tea/best/best_checkpoint.pth").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
