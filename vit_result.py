# 导入所需的库
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import timm

class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# 数据预处理 - 包括调整图像大小
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    # 初始化 TensorBoard
    writer = SummaryWriter("runs/cifar10_vit_experiment")

    # 设置训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 CIFAR-10 数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=6,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=6,
                                             shuffle=True, num_workers=2)

    # Define the Vision Transformer model
    net = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    net = net.to(device)
    # 初始化网络、损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # 将模型结构添加到 TensorBoard
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    writer.add_graph(net, images.to(device))

    # 训练网络
    for epoch in range(200):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
                running_loss = 0.0

    # 测试网络
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 添加到总的标签和预测列表中
            all_labels.extend(labels.cpu().view(-1))
            all_predictions.extend(predicted.cpu().view(-1))

    # 将测试图像及其标签添加到 TensorBoard
    writer.add_image('Test Images', torchvision.utils.make_grid(images.cpu()))

    # 将总的标签和预测添加为 PR 曲线
    writer.add_pr_curve('PR Curve', torch.tensor(all_labels), torch.tensor(all_predictions))

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
    writer.add_scalar('accuracy', 100 * correct / total, 0)

    writer.close()

if __name__ == '__main__':
    main()
