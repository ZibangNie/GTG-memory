import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_runtime import read_nonempty_lines


class RuntimeCheckTests(unittest.TestCase):
    def test_reads_utf8_bom_split_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            split_path = Path(temp_dir) / "training.txt"
            split_path.write_text("\ufeffvideo1\n\nvideo2\n", encoding="utf-8")

            self.assertEqual(read_nonempty_lines(split_path), ["video1", "video2"])


if __name__ == "__main__":
    unittest.main()
