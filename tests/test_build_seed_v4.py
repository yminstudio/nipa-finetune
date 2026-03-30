import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "scripts/gen_dataset_v4/06_build_seed_v4.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_seed = load_module("build_seed_v4", BUILD_PATH)


class BuildSeedV4Tests(unittest.TestCase):
    def test_build_record_supports_custom_dataset_version(self):
        record = build_seed.build_record(
            1,
            {
                "question": "질문",
                "answer": "답변",
                "review_status": "pass",
                "chapter": "2. 펀드평가 방법론",
            },
            dataset_version="v4_1",
            dataset_prefix="zeroin.seed_v4_1",
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["id"], "zeroin.seed_v4_1_0001")
        self.assertEqual(record["meta"]["dataset_version"], "v4_1")

    def test_resolve_output_paths_supports_custom_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ch01_path, ch02_path = build_seed.resolve_output_paths(
                Path(tmpdir),
                chapter_output_prefix="seed_v4_1",
            )

        self.assertEqual(ch01_path.name, "seed_v4_1_ch01.jsonl")
        self.assertEqual(ch02_path.name, "seed_v4_1_ch02.jsonl")


if __name__ == "__main__":
    unittest.main()
