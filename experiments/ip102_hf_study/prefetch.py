"""Download and verify all three selected Hugging Face pretrained checkpoints.

Files go to the user's normal Hugging Face cache, never into git.
Run: python experiments/ip102_hf_study/prefetch.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download
from experiments.ip102_hf_study.train import MODELS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("all", *MODELS), default="all")
    p.add_argument("--revision", default="main")
    args = p.parse_args()
    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        repo_id = MODELS[name]
        print(f"\nDownloading {name}: https://huggingface.co/{repo_id}", flush=True)
        for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
            path = Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                        revision=args.revision))
            print(f"Verified {filename}: {path.stat().st_size / 1024**2:.1f} MiB cached",
                  flush=True)
    print("\nAll requested pretrained files are cached. No IP102 training has started.")


if __name__ == "__main__":
    main()
