#!/usr/bin/env python3
"""Verify source provenance, artifact bytes, and complete media decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, text: bool = True) -> str | bytes:
    output = subprocess.check_output(["git", *args], text=text)
    return output.strip() if text else output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("docs/demo/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source_sha = manifest["source_sha"]
    assert git("show", "-s", "--format=%T", source_sha) == manifest["source_tree"]

    for record in manifest["runtime_inputs"]:
        data = git("show", f"{source_sha}:{record['path']}", text=False)
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert git("rev-parse", f"{source_sha}:{record['path']}") == record["git_blob"]

    for record in manifest["artifacts"]:
        path = Path(record["path"])
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]
        assert git("hash-object", str(path)) == record["git_blob"]
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            check=True,
        )

    receipt = manifest["smoke_receipt"]
    assert receipt["evaluation_examples"] == 9
    assert receipt["evaluation_batches"] == 2
    assert receipt["evaluation_tail_included"] is True
    assert receipt["checkpoint_parameter_roundtrip"] is True
    print(json.dumps({"verified": True, "source_sha": source_sha,
                      "source_tree": manifest["source_tree"],
                      "artifacts": len(manifest["artifacts"]),
                      "evaluation_examples": receipt["evaluation_examples"]}, indent=2))


if __name__ == "__main__":
    main()
