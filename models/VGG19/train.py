"""Train VGG19; invoke from repository root: python models/VGG19/train.py ..."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.train import main

if __name__ == "__main__":
    main("vgg19")
