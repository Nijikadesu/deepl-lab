import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        """=============== Your code below ================="""
        # make sure your layers use the default dtype (aka torch.float32)
        
        # Conv1: 1 input channel, 6 output channels, 5x5 kernel size
        # Conv2: 6 input channels, 16 output channels, 5x5 kernel size
        # FC1: 16*4*4 input features, 120 output features
        # FC2: 120 input features, 84 output features
        # FC3: 84 input features, 10 output features

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.ReLU1 = nn.ReLU()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.Conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.ReLU2 = nn.ReLU()
        self.MaxPool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.FC1 = nn.Linear(16 * 4 * 4, 120)
        self.ReLU3 = nn.ReLU()
        self.FC2 = nn.Linear(120, 84)
        self.ReLU4 = nn.ReLU()
        self.FC3 = nn.Linear(84, 10)

        """=============== Your code above ================="""

    def forward(self, x):

        """=============== Your code below ================="""
        # Conv1 -> ReLU -> MaxPool2d(2x2) -> Conv2 -> ReLU -> MaxPool2d(2x2) -> Resize -> FC1 -> ReLU -> FC2 -> ReLU -> FC3

        x = self.MaxPool1(self.ReLU1(self.Conv1(x)))
        x = self.MaxPool2(self.ReLU2(self.Conv2(x)))
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1))
        x = self.FC3(self.ReLU4(self.FC2(self.ReLU3(self.FC1(x)))))

        """=============== Your code above ================="""
        return x


if __name__ == "__main__":
    from check import validate_model
    validate_model(LeNet)