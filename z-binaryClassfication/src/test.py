# coding=gb2312
import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import metrics
from tqdm import tqdm
import dataset
from model_utils import resnet18  # 根据实际使用的模型导入

'''
    test.py可修改的部分：
        1、from model_utils import resnet18  # import的模型可根据实际修改
        2、parser.add_argument命令行参数，只可修改 default 字段
        3、main函数部分的模型调用：model = resnet18(num_classes=2, include_top=True)  # 二分类任务 num_classes=2
        4、数据加载部分，image_datasets和test_dataset
'''

parser = argparse.ArgumentParser(description='Classification Competition Project')
parser.add_argument('--name', default='binary_resnet', type=str, help='name of the run')
parser.add_argument('--seed', default=2, type=int, help='random seed')
parser.add_argument('--test',
                    default='',
                    type=str, help='path to test image files/labels')
parser.add_argument('--sep', default=',', type=str, help='column separator used in csv (default: ",")')
parser.add_argument('--data_dir', default=r'D:\Desktop\classification(1)\classification\data\MRI\test',
                    type=str, help='root directory of images')
parser.add_argument('--best_state_path',
                    default='',
                    type=str, help='path to load best state')
parser.add_argument('--batch_size', default=32, type=int, help='batch size (default: 32)')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def Singletask_test_model_save_results(model, dataloader, device=None):
    print("Testing the model and saving the results:", flush=True)

    model.eval()
    preds_ = []
    labels_ = []

    running_corrects = 0
    count = 0
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test Iteration"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)

            preds_.extend(preds.cpu().numpy().tolist())
            labels_.extend(labels.data.cpu().numpy().tolist())
            running_corrects += torch.sum(preds == labels.data)
            count += len(labels.data)

    f1_score = metrics.f1_score(labels_, preds_, average="macro")
    print(f"F1 Score: {f1_score}")

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    file_name = f"{parent_dir}/out/single_output.txt"
    with open(file_name, "a") as file:
        file.write(str(f1_score) + "\n")
    print("测试结果已保存到", file_name)


def main():
    global device
    use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(use_gpu)
    print("Using device: {}".format(use_gpu), flush=True)
    args = parser.parse_args()
    print(args, flush=True)
    set_seed(args.seed)

    data_dir = args.data_dir
    best_state_path = args.best_state_path
    batch_size = args.batch_size
    print(best_state_path)
    # 加载模型，适用于二分类任务
    model = resnet18(num_classes=2, include_top=True)
    best_state = torch.load(best_state_path)
    model.load_state_dict(best_state)

    # 数据预处理
    data_transforms = {
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.CenterCrop((256, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("Initializing Datasets and Dataloaders...", flush=True)
    test_dataset = dataset.BinaryClassificationDataset(args.test, sep=args.sep, root_dir=data_dir, transform=data_transforms['test'])
    print("There are {} test images.".format(len(test_dataset)))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()), flush=True)
        model = torch.nn.DataParallel(model)

    Singletask_test_model_save_results(model, test_dataloader, device=device)


if __name__ == "__main__":
    main()
