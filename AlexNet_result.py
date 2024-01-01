import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F


def show_images_with_labels(images, true_labels, pred_labels, writer, step, title):
    """
    Show images with true and predicted labels and write to TensorBoard.

    Args:
        images (tensor): Batch of images.
        true_labels (tensor): Actual labels of the batch.
        pred_labels (tensor): Predicted labels of the batch.
        writer (SummaryWriter): TensorBoard writer.
        step (int): Current step for TensorBoard.
        title (str): Title for the image grid.
    """
    # Unnormalize the images
    images = images / 2 + 0.5

    # Convert tensor to numpy for plotting
    img_grid = torchvision.utils.make_grid(images)

    # Write to TensorBoard
    writer.add_image(title, img_grid, step)


# Define the AlexNet architecture for CIFAR-10
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Initialize TensorBoard writer
writer = SummaryWriter("runs/AlexNet_CIFAR10")

# Define data transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 加载CIFAR-10数据集
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 初始化TensorBoard
writer = SummaryWriter("runs/AlexNet_CIFAR10")

# 训练循环
for epoch in range(10):  # 根据需要调整epoch数量
    net.train()
    running_loss = 0.0
    total_samples = 0
    correct_samples = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_samples += (predicted == labels).sum().item()

    # 每个epoch后记录损失和准确率
    writer.add_scalar('Training Loss', running_loss / len(trainloader), epoch)
    writer.add_scalar('Training Accuracy', correct_samples / total_samples, epoch)

# 评估
net.eval()
correct = 0
total = 0
class_preds = []
class_labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 为PR曲线收集数据
        class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
        class_preds.append(class_probs_batch)
        class_labels.append(labels)

accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')

# 添加测试数据到TensorBoard
class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
class_labels = torch.cat(class_labels)
for i in range(10):
    labels_i = class_labels == i
    preds_i = class_preds[:, i]
    writer.add_pr_curve(f'class_{i}', labels_i, preds_i, global_step=0)

# 关闭TensorBoard
writer.close()