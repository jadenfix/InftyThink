#!/usr/bin/env python3
"""Capture a truthful terminal-style demo from the real tiny JAX smoke receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


RUNTIME_INPUTS = [
    "scripts/capture_demo.py",
    "src/model/config.py",
    "src/model/transformer.py",
    "src/training/checkpointer.py",
    "src/training/losses.py",
    "src/training/lr_schedule.py",
    "src/training/smoke.py",
    "src/training/trainer.py",
    "requirements-smoke.in",
    "requirements-smoke.txt",
]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, **kwargs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, text: bool = True) -> str | bytes:
    output = subprocess.check_output(["git", *args], text=text)
    return output.strip() if text else output


def escape_ass(value: str) -> str:
    # Preserve the intentional ASS ``\N`` line-break commands constructed by
    # ``write_ass`` while preventing visible text from opening style overrides.
    return value.replace("{", r"\{").replace("}", r"\}")


def write_ass(path: Path, receipt: dict, source_sha: str) -> None:
    loss_before, loss_after = receipt["losses"]
    screens = [
        (
            "0:00:00.00",
            "0:00:03.20",
            "SOURCE-PINNED INTEGRITY DEMO\\N\\N"
            f"source  {source_sha[:12]}\\N"
            "scope   synthetic tokens + randomly initialized tiny CPU model\\N\\N"
            "No math accuracy, paper reproduction, or performance claim",
        ),
        (
            "0:00:03.20",
            "0:00:06.40",
            "REAL JAX CPU TRAINING\\N\\N"
            f"optimizer steps       {receipt['steps']}\\N"
            f"loss step 1           {loss_before:.6f}\\N"
            f"loss step 2           {loss_after:.6f}\\N"
            f"parameter delta L1    {receipt['parameter_delta_l1']:.6f}\\N\\N"
            "finite losses + parameters changed",
        ),
        (
            "0:00:06.40",
            "0:00:09.60",
            "CHECKPOINT + GENERATION\\N\\N"
            f"parameter roundtrip   {str(receipt['checkpoint_parameter_roundtrip']).lower()}\\N"
            f"generated tokens      {receipt['generated_tokens']}\\N"
            f"generated shape       {receipt['generated_shape']}\\N\\N"
            "restored checkpoint used for greedy generation",
        ),
        (
            "0:00:09.60",
            "0:00:12.80",
            "EVALUATION TAIL PROOF\\N\\N"
            f"examples evaluated    {receipt['evaluation_examples']} / 9\\N"
            f"batches evaluated     {receipt['evaluation_batches']}  (8 + 1)\\N"
            f"tail included         {str(receipt['evaluation_tail_included']).lower()}\\N"
            f"equal-task eval loss {receipt['evaluation_loss']:.6f}\\N\\N"
            "every synthetic evaluation example counted",
        ),
    ]
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1440
PlayResY: 900

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Terminal,DejaVu Sans Mono,30,&H00E8F2FF,&H000000FF,&H00121C2B,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,145,100,210,1
Style: Header,DejaVu Sans,22,&H0062E6B8,&H000000FF,&H00121C2B,&H00000000,1,0,0,0,100,100,2,0,1,0,0,7,145,100,118,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:12.80,Header,,0,0,0,,INFTYTHINK  /  VERIFIED SYNTHETIC CPU WORKFLOW
"""
    dialogue = "".join(
        f"Dialogue: 0,{start},{end},Terminal,,0,0,0,,{escape_ass(text)}\n"
        for start, end, text in screens
    )
    path.write_text(header + dialogue)


def artifact_record(path: Path) -> dict:
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-count_frames",
                "-show_entries", "stream=codec_name,width,height,nb_read_frames,duration",
                "-show_entries", "format=duration", "-of", "json", str(path),
            ],
            text=True,
        )
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "git_blob": git("hash-object", str(path)),
        "probe": probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/demo"))
    args = parser.parse_args()

    source_sha = git("rev-parse", args.source_sha)
    source_tree = git("show", "-s", "--format=%T", source_sha)
    if git("status", "--porcelain"):
        raise RuntimeError("Capture requires a clean source checkout")
    if git("rev-parse", "HEAD") != source_sha:
        raise RuntimeError("HEAD must equal --source-sha during capture")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="inftythink-demo-") as tmp:
        tmp_path = Path(tmp)
        smoke_dir = tmp_path / "smoke"
        env = os.environ.copy()
        env.update(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        completed = run(
            [args.python, "-m", "src.training.smoke", "--output", str(smoke_dir)],
            capture_output=True,
            env=env,
        )
        receipt = json.loads((smoke_dir / "smoke.json").read_text())
        ass_path = tmp_path / "demo.ass"
        write_ass(ass_path, receipt, source_sha)

        mp4 = args.output_dir / "inftythink-evaluation-integrity.mp4"
        gif = args.output_dir / "inftythink-evaluation-integrity.gif"
        poster = args.output_dir / "inftythink-evaluation-integrity.png"
        common_filter = (
            "drawbox=x=70:y=70:w=1300:h=760:color=0x101827@0.97:t=fill,"
            "drawbox=x=70:y=70:w=8:h=760:color=0x62e6b8:t=fill,"
            f"subtitles={ass_path}"
        )
        run([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            "color=c=0x07111f:s=1440x900:r=30:d=12.8",
            "-vf", common_filter, "-c:v", "libx264", "-preset", "slow",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4),
        ])
        run([
            "ffmpeg", "-y", "-v", "error", "-ss", "10.5", "-i", str(mp4),
            "-frames:v", "1", str(poster),
        ])
        run([
            "ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vf",
            "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            "-loop", "0", str(gif),
        ])

    source_files = []
    for path in RUNTIME_INPUTS:
        data = git("show", f"{source_sha}:{path}", text=False)
        source_files.append({
            "path": path,
            "git_blob": git("rev-parse", f"{source_sha}:{path}"),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {
        "scope": receipt["scope"],
        "claims_excluded": [
            "math accuracy", "paper reproduction", "model quality", "performance",
            "live data", "provider integration",
        ],
        "source_sha": source_sha,
        "source_tree": source_tree,
        "capture_command": (
            "JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
            f"{args.python} scripts/capture_demo.py --source-sha {source_sha} "
            f"--python {args.python} --output-dir {args.output_dir}"
        ),
        "python": platform.python_version(),
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        "smoke_stdout": completed.stdout,
        "smoke_receipt": receipt,
        "runtime_inputs": source_files,
        "artifacts": [artifact_record(path) for path in (gif, mp4, poster)],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"source_sha": source_sha, "source_tree": source_tree,
                      "receipt": receipt, "artifacts": manifest["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
