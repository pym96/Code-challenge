import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

# 定义简单的卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# 数据预处理：将图像转换为PyTorch张量并标准化
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def main():
    # 设定 TensorBoard
    writer = SummaryWriter("runs/cifar10_conv_experiment")

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=6,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=6,
                                            shuffle=True, num_workers=2)
    
    examples = iter(testloader)
    example_data, example_targets = next(examples)

    ############## TENSORBOARD ########################
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('mnist_images', img_grid)

    # 显示6个图像
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        img = example_data[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img / 2 + 0.5  # 重新缩放图像数据到 [0, 1]
        plt.imshow(img)
        
    net = Net().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    writer.add_graph(net, example_data.to(device))


    running_correct = 0
    running_loss = 0.0
    n_total_steps = len(trainloader)
    # 训练网络并记录到 TensorBoard
    for epoch in range(100):  
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            if i % 100 == 0:    # 每 1000 批次记录一次
                writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
                running_accuracy = running_correct / 100 / predicted.size(0)
                writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
                running_correct = 0
                running_loss = 0.0

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    class_labels = []
    class_preds = []

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in testloader:
            labels = labels.to(device)
            outputs = net(images)
            # max returns (value ,index)
            values, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

            class_preds.append(class_probs_batch)
            class_labels.append(labels)


        class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
        class_labels = torch.cat(class_labels)

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')

        ############## TENSORBOARD ########################
        classes = range(10)
        for i in classes:
            labels_i = class_labels == i
            preds_i = class_preds[:, i]
            writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
            writer.close()

    print('Finished Training')

if __name__ == '__main__':
    main()
