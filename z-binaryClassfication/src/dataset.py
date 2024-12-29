# coding=gb2312
import os
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
    dataset.pyΪ���������ݶ�ȡģ�ͣ��Ͻ��޸ģ�ֻ�����
'''


class BinaryClassificationDataset(Dataset):

    def __init__(self, file_path, sep, root_dir, transform=None):
        self.file_path = file_path
        self.root_dir = root_dir
        self.transform = transform

        # ��ȡCSV�ļ����������'image_path'��'label'����
        df = pd.read_csv(file_path, sep=sep, na_filter=False)

        # ��ȡͼ��·���ͱ�ǩ
        self.X = df['image_path'].tolist()
        self.y = df['label'].tolist()

        # ����ǩת��Ϊ������ʽ��0��1��
        self.classes, self.class_to_idx = self._find_classes()
        # �����������ݷ����Ӧ
        self.samples = list(zip(self.X, [self.class_to_idx[label] for label in self.y]))

    def __getitem__(self, index):
        path, label = self.samples[index]
        img_path = os.path.join(self.root_dir, path)

        with open(img_path, 'rb') as f:
            img = Image.open(f)  #ת��Ϊ��ͨ��  ������ѵ��
            if img.mode != 'RGB':  
                img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def _find_classes(self):
        # ȷ����ǩΪ0��1�Ķ�����
        classes_set = set(self.y)
        classes = sorted(list(classes_set))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
