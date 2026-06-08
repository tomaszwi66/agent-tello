"""Download a local faster-whisper model for offline English voice commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="h2oai/faster-whisper-large-v3-turbo")
    ap.add_argument("--out", default="models/whisper-large-v3-turbo")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.repo} -> {out}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print(f"ready: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
