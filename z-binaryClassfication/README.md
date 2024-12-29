
# 比赛框架使用说明
## 一、文件结构
```
competition_frame/
│
└── classification/
    ├── data                # 分类数据集
    │   └── MRI_data/       # MRI 图像数据集
    │       └── MRI/
    │           ├── train/          # 训练数据集
    │           │   ├── AD/         # 轻度病变
    │           │   └── NC/         # 没有病变
    │           │   └── test.csv    # 测试图片地址文件
    │           └── test/           # 测试数据集
    │               ├── AD/         # 轻度病变
    │               └── NC/         # 没有病变
    |               └── train.csv   # 训练图片地址文件  
    │                 
    |           
    ├── out                # 结果保存文件
    │
    └── src                # 脚本文件
        ├── dataset.py       # 数据读取脚本
        ├── model_utils.py   # 模型配置脚本
        ├── test.py          # 测试脚本
        └── train.py         # 训练脚本
```
## 二、模型训练和测试权重文件
为了实现标准化统一测试，统一规定模型最优权重文件命名为：`xxx_best_state.pth`。
- 训练过程保存权重文件：
  ```python
  torch.save(model.state_dict(), 'best_state.pth')
  ```
- 测试过程调用权重文件：
  ```python
  best_state = torch.load(best_state_path)
  model.load_state_dict(best_state)
  ```
## 三、classification 接口
### 1. dataset.py
该文件为完整的数据读取脚本，只允许调用，严禁修改。
MRI 图像分类任务数据读取类（CLASSMRI）调用示例：
```python
train_dataset = CLASSMRI(train_image_paths, train_labels, data_transforms['train'])
val_dataset = CLASSMRI(val_image_paths, val_labels, data_transforms['val'])
```
DataLoader 示例：
```python
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
```
### 2. CSV 文件结构
数据集未提供合并的 CSV 文件，请根据以下格式自定义路径和标签。
### 3. train.py
该文件已提供所有可调用的包，严禁增添或删减，只允许调用。模型的训练流程包括数据加载、损失函数、优化器以及训练过程中权重保存，确保运行的完整性。
训练时可以使用以下示例命令：
```bash
python src/train.py --epochs 50 --batch_size 32 --lr 0.001
```
其中train_csv_path = ''为训练集的地址文件，train_dataset里面的root_dir=''为训练图片存放位置要与csv文件想结合
### 4. model_utils.py
该文件已提供所有可调用的包，严禁增添或删减，只允许调用。用于模型的定义、配置和初始化。
### 5. test.py
该文件为完整的测试脚本，适用于 MRI 图像分类任务学习，只需要更改其文件存放地址
测试时的调用示例：
```bash
python src/test.py --model_path 'best_state.pth' --test_data 'data/MRI_data/MRI/test'
```
测试过程中会加载最优权重文件并生成模型在测试集上的结果。
