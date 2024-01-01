import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Initialize TensorBoard
writer = SummaryWriter("runs/resnet_cifar10_experiment")

# Define the ResNet model
# Define the ResNet model
# Use ResNet18_Weights.DEFAULT instead of pretrained=True
weights = ResNet18_Weights.DEFAULT if torch.cuda.is_available() else None
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10
model = model.to(device)

# Add model graph to TensorBoard
images, _ = next(iter(trainloader))
writer.add_graph(model, images.to(device))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
            running_loss = 0.0

# Test the model
correct = 0
total = 0
class_labels = []
class_preds = []
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        class_labels.extend(labels.view(-1).cpu())
        class_preds.extend(outputs.softmax(dim=1).cpu())

accuracy = 100 * correct / total
print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)

# Log accuracy to TensorBoard
writer.add_scalar('accuracy', accuracy, 0)

# Log PR curve
for i in range(10):
    labels_i = torch.tensor(class_labels) == i
    preds_i = torch.tensor([class_pred[i] for class_pred in class_preds])
    writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)

writer.close()
