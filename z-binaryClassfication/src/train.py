import os
import torchvision
from torch import optim
from torch.utils.data import DataLoader
import dataset
import torch
import torch.nn as nn
from model_utils import resnet18  # 确保该模型支持二分类输出

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 定义训练数据集路径
    # currentpath = os.path.dirname(os.path.abspath(__file__))
    train_csv_path = 'data/MRI/train/train.csv'

    # 定义数据预处理的转换方法
    data_transforms = {
        'train': torchvision.transforms.Compose([
            # torchvision.transforms.Resize((256, 224)),  调整输入
            torchvision.transforms.Resize((224, 224)), #将图像调整为 224x224 大小。
            torchvision.transforms.RandomHorizontalFlip(), #随机旋转图像，旋转角度为 10°。
            torchvision.transforms.RandomRotation(10), #随机水平翻转图像，翻转概率为 50%。
            torchvision.transforms.ToTensor(), #将图像转换为张量。
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            #对图像进行标准化。   如果原始像素值是 [0, 255]（整数），会被转换为浮点数 [0.0, 1.0]。
            # ResNet 模型的预训练权重是在 ImageNet 数据集上训练的。
            #ImageNet 数据集的像素值分布大约是 mean=[0.485, 0.456, 0.406] 
            #和 std=[0.229, 0.224, 0.225]，归一化能让输入分布与预训练时一致，提升模型效果。
        ]),  
    }
    '''
    Compose: torchvision.transforms.Compose 
    是一个工具，可以将多个转换操作串联起来，形成一个管道，按顺序对输入数据进行处理。
    '''


    root_dir = 'data/MRI/train/'
    # 初始化训练数据集和数据加载器
    train_dataset = dataset.BinaryClassificationDataset(train_csv_path, sep=',', root_dir=root_dir,
                                                        transform=data_transforms['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    # 初始化ResNet18模型，二分类任务的输出维度为2
    model = resnet18(num_classes=2, include_top=True).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = corrects.double() / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), '../out/Binary_ResNet18_best_state_epoch=20_batch_size_32.pth')


if __name__ == '__main__':
    main()
