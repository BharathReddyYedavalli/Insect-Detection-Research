"""Train ResNet18; invoke from repository root: python models/ResNet18/train.py ..."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.train import main

if __name__ == "__main__":
    main("resnet18")
