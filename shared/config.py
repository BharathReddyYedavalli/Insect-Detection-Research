"""Config inheritance, seeds and per-run provenance."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json
import platform
import random
import subprocess
import sys
import yaml
import numpy as np
import torch
import torchvision

PROJECT = Path(__file__).resolve().parents[1]
DISPLAY = {
    "resnet18": "ResNet18", "resnet50": "ResNet50",
    "vgg16": "VGG16", "vgg19": "VGG19",
}


def merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else deepcopy(v)
    return out


def config_for(name: str) -> dict:
    with open(PROJECT / "shared" / "default.yaml", encoding="utf-8") as fh:
        defaults = yaml.safe_load(fh)
    with open(PROJECT / "models" / DISPLAY[name] / "config.yaml", encoding="utf-8") as fh:
        changes = yaml.safe_load(fh)
    config = merge(defaults, changes or {})
    config["model"] = name
    return config


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # warn_only handles operations without a deterministic implementation.
        torch.use_deterministic_algorithms(True, warn_only=True)


def new_run(name: str):
    root = PROJECT / "models" / DISPLAY[name] / "runs"
    path = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_config(config: dict, run: Path):
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    flat = []
    for section, values in config.items():
        if isinstance(values, dict):
            flat += [f"{section}.{key}: {val}" for key, val in values.items()]
        else:
            flat.append(f"{section}: {values}")
    (run / "parameters.txt").write_text("\n".join(flat) + "\n", encoding="utf-8")


def capture_environment(run: Path):
    def version(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5,
                                           text=True).strip()
        except Exception:
            return None
    env = {
        "python": sys.version, "platform": platform.platform(),
        "pytorch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": version(["git", "rev-parse", "HEAD"]),
    }
    (run / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
