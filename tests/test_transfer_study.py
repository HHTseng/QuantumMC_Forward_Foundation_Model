"""Re-entering a completed study must preserve its execution provenance."""

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/run_transfer_study.py"
SPEC = importlib.util.spec_from_file_location("run_transfer_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_noop_resume_preserves_existing_execution_record(tmp_path):
    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    content = b"immutable test input"
    (inputs / "sample.parquet").write_bytes(content)
    manifest = {"datasets": {"sample": {
        "file": "sample.parquet", "sha256": hashlib.sha256(content).hexdigest()}}}
    (inputs / "input_manifest.json").write_text(json.dumps(manifest))
    original = '{"commands":[{"seconds":12.5}],"finished_utc":"original"}\n'
    (output / "execution.json").write_text(original)
    arguments = [str(SCRIPT), "--inputs", str(inputs), "--output", str(output),
                 "--skip-response", "--skip-efficiency"]
    with patch("sys.argv", arguments):
        study.main()
    assert (output / "execution.json").read_text() == original
    (inputs / "sample.parquet").write_bytes(b"changed input")
    with patch("sys.argv", arguments), pytest.raises(ValueError, match="checksum mismatch"):
        study.main()
