import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

"""=============== Your code below ================="""
# Initialize the LeNet model, loss function, and optimizer
from model import LeNet
net = LeNet().to(device) # LeNet
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss
optimizer = optim.SGD(net.parameters(), lr=0.1) # SGD
"""=============== Your code above ================="""

# Train the model
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        """=============== Your code below ================="""
        # calculate the loss
        loss = criterion(outputs, labels)

        # backpropagation
        loss.backward()

        # update the weights
        optimizer.step()
        """=============== Your code above ================="""

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch: {epoch+1:3d}, Batch: {i+1:5d}, Loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1) # return max values and corresponding indices
        total += labels.size(0)

        """=============== Your code below ================="""
        # calculate the number of correct predictions
        correct += (labels == predicted).sum().item()
        """=============== Your code above ================="""

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")
