import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Simple convolutional neural network for image classification
    on grayscale 28x28 datasets (e.g., MNIST or FashionMNIST).

    Architecture:
        - Conv2d(1 → 16, kernel_size=3, padding=1)
        - ReLU
        - Conv2d(16 → 32, kernel_size=3, padding=1)
        - ReLU
        - MaxPool2d(kernel_size=2)
        - Flatten
        - Linear(32 * 14 * 14 → 10)
        - ReLU
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                              # 14x14
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x)) # comment this 
        return x