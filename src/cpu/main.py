"""
Main script for training and testing a model on a specified dataset (MNIST).

Key functionalities:
- Initializes random number generators to ensure reproducibility.
- Sets up DataLoaders for training and testing with:
    - batch_size = 64
    - shuffle enabled only for training
    - num_workers > 0 to allow parallel data loading
    - pin_memory=True to speed up transfer of batches from CPU to GPU
- Supports execution on both CPU and GPU.
- Measures and prints total training and testing time, using torch.cuda.synchronize() for accurate GPU timing.
- Initializes Trainer and Tester consistent with the DataLoaders.
- Computes and prints average loss and accuracy during testing.

DataLoader considerations:
- `num_workers > 0` enables parallel data loading, improving performance especially on GPU.
- `pin_memory=True` allocates batch memory as "pinned", allowing faster asynchronous transfer to GPU.
- These settings improve training and testing efficiency on GPU without requiring changes to the training loop.

Usage:
    python main.py --device [cuda, cpu] --dataset [mnist, fashion] --pin [True, False] ---numw ["0", "1", "2", "4", "8"]
"""

import time
import torch
import argparse
import torch
import random
import numpy as np
import os

from model import Net
from train import Trainer
from test import Tester
from dataloader import get_dataloader


def main(args):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # training options
    device = args.device
    epoch = 5
    lr = 1e-2

    # Dataloader 
    train_loader = get_dataloader(dataset_name= args.dataset, train=True, num_workers=args.numw, pin_memory= args.pin)
    test_loader  = get_dataloader(dataset_name=args.dataset, train=False, num_workers=args.numw, pin_memory= args.pin)
  

    # Initialize trainer, tester and model 
    trainer = Trainer(epochs= epoch, lr= lr, train_loader= train_loader, device= device)
    tester = Tester(test_loader= test_loader, device= device)
    model = Net()

    # chrono training 
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    model = trainer.train(model=model)

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Time for training {epoch} is: {end_time - start_time:.3f} s")

    # chrono testing
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    tester.test(model= model)

    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    print(f"Time for testing is: {end_time - start_time:.3f} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices= ["cuda", "cpu"], default = "cpu")
    parser.add_argument("--dataset", choices= ["mnist", "fashion"], default = "mnist")
    parser.add_argument("--pin", choices= ["True", "False"], default = "True")
    parser.add_argument("--numw", choices= ["0", "1", "2", "4", "8"], default = 4)

    args = parser.parse_args()
    main(args)