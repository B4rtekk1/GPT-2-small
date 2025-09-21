import random
import torch
import numpy as np
import os

def setSeed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def logs_setup(path: str | None):
    if not path:
        print("Logging path not set")
    else:
        path = os.path.join('logs', path)
        os.mkdir(path)
        print(f"Logging path: {path}")