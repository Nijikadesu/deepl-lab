import torch
import torch.nn as nn


"""fully connected layer"""
fc_layer = nn.Linear(10, 5)
fc_input = torch.randn(1, 10)
fc_output = fc_layer(fc_input)
print("Layer       :", fc_layer)
print("Input shape :", fc_input.shape)
print("Output shape:", fc_output.shape)
print()


"""convolutional layer"""
conv_layer = nn.Conv2d(1, 3, kernel_size=3)
conv_input = torch.randn(1, 1, 5, 5)
conv_output = conv_layer(conv_input)
print("Layer       :", conv_layer)
print("Input shape :", conv_input.shape)
print("Output shape:", conv_output.shape)
print()


"""max pooling layer"""
pool_layer = nn.MaxPool2d(kernel_size=2)
pool_input = torch.randn(1, 1, 4, 4)
pool_output = pool_layer(pool_input)
print("Layer       :", pool_layer)
print("Input shape :", pool_input.shape)
print("Output shape:", pool_output.shape)
print()


"""activation function"""
activation = nn.ReLU()
activation_input = torch.randn(1, 5)
activation_output = activation(activation_input)
print("Layer :", activation)
print("Input :", activation_input)
print("Output:", activation_output)
print()


"""loss function"""
loss_func = nn.CrossEntropyLoss()
output = torch.randn(1, 5)
target = torch.empty(1, dtype=torch.long).random_(5)
loss = loss_func(output, target)
print("Layer     :", loss_func)
print("Prediction:", output)
print("Target    :", target)
print("Loss      :", loss)
print("Loss(raw) :", (-torch.log(torch.softmax(output, dim=1)[0][target])).mean())
print()


class SimpleNN(nn.Module):
    """simple neural network"""

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(3 * 13 * 13, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 3 * 13 * 13)
        x = self.fc(x)
        return x


net = SimpleNN()
input = torch.randn(1, 28, 28)
output = net(input)

print("SimpleNN:", SimpleNN())
print("Input   :", input.shape)
print("Output  :", output.shape)

