from torchvision import datasets, transforms
import torch 

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST
}

def get_dataloader(num_workers, dataset_name, train=True, batch_size=64, pin_memory=True):
    """
    Load a PyTorch DataLoader for the specified dataset.

    This function retrieves either the training or test split of a supported
    torchvision dataset (e.g., MNIST or FashionMNIST), applies a basic
    tensor transformation, and wraps it into a DataLoader ready for training
    or evaluation.

    Args:
        dataset_name (str): Name of the dataset to load. Supported values
            are defined in the DATASETS dictionary (e.g., "mnist", "fashion").
        train (bool, optional): If True, loads the training split.
            If False, loads the test split. Defaults to True.
        batch_size (int, optional): Number of samples per batch.
            Defaults to 64.
        num_workers (int, optional): Number of subprocesses used for
            data loading. Higher values may improve performance depending
            on the system. Defaults to 2.
        pin_memory (bool, optional): If True, the DataLoader will copy
            tensors into CUDA pinned memory before returning them.
            This can improve GPU transfer speed when using CUDA.
            Defaults to True.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance for the
        requested dataset split.
    """
    dataset_cls = DATASETS.get(dataset_name.lower())
    
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset {dataset_name}")

    dataset = dataset_cls(root="./datasets", train=train, download=True, transform=transforms.ToTensor())

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers= int(num_workers), pin_memory=pin_memory)
    
    return loader
