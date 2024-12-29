# coding=gb2312
import os
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
    dataset.py为完整的数据读取模型，严禁修改，只许调用
'''


class BinaryClassificationDataset(Dataset):

    def __init__(self, file_path, sep, root_dir, transform=None):
        self.file_path = file_path
        self.root_dir = root_dir
        self.transform = transform

        # 读取CSV文件，假设包含'image_path'和'label'两列
        df = pd.read_csv(file_path, sep=sep, na_filter=False)

        # 读取图像路径和标签
        self.X = df['image_path'].tolist()
        self.y = df['label'].tolist()

        # 将标签转换为数字形式（0和1）
        self.classes, self.class_to_idx = self._find_classes()
        # 将索引与数据分类对应
        self.samples = list(zip(self.X, [self.class_to_idx[label] for label in self.y]))

    def __getitem__(self, index):
        path, label = self.samples[index]
        img_path = os.path.join(self.root_dir, path)

        with open(img_path, 'rb') as f:
            img = Image.open(f)  #转化为三通道  网络中训练
            if img.mode != 'RGB':  
                img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def _find_classes(self):
        # 确保标签为0和1的二分类
        classes_set = set(self.y)
        classes = sorted(list(classes_set))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
