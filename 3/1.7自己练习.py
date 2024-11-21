import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
# from d2l import torch as d2l

#     预测1/jpg 和 2.jpg的效果不是很好

# Step 1: 加载图片并预处理
def preprocess_image(img_path):
    """
    预处理输入图片：
    1. 转换为灰度图像
    2. 调整为 28x28 尺寸
    3. 转换为 PyTorch Tensor，形状为 (1, 1, 28, 28)
    """
    # 加载图片
    img = Image.open(img_path).convert('L')  # 转为灰度图像
    img = img.resize((28, 28))  # 调整为 28x28 像素
    
    # 可视化图片（可选）
    plt.imshow(img, cmap='gray')
    plt.title("Processed Image")
    plt.show()
    
    # 转换为 PyTorch Tensor，形状 (1, 1, 28, 28)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(0.5, 0.5)  # 归一化到 [-1, 1]（可选）
    ])
    img_tensor = transform(img).unsqueeze(0)  # 增加 batch 维度
    return img_tensor

# Step 2: 使用模型进行预测
def predict_image(net, img_tensor):
    """
    使用训练好的模型对图片进行预测
    """
    net.eval()  # 切换模型到评估模式
    with torch.no_grad():  # 关闭梯度计算
        output = net(img_tensor)  # 前向传播
        predicted_class = torch.argmax(output, axis=1).item()  # 获取预测类别
    return predicted_class

'''
    #  preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))  这是3.7节预测时的代码，可以对比学习一下   
    这里的代码会手动切换到评估模式， 并且在预测时关闭梯度计算，

    d2l的那种写法却没有那样做。 至于预测几种的话，是可以进行调整的，   预测一批还是预测单独几张

'''

# Step 3: 加载和使用模型

net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 10)
)
net.load_state_dict(torch.load('3.7model.pth', weights_only=True))  # 加载训练好的模型

# 输入图片路径
img_path = "2.jpg"  # 替换为你图片的路径

# 预处理图片
img_tensor = preprocess_image(img_path)

# 显示预测结果和图片
def display_prediction(img_tensor, predicted_class, labels):
    """
    显示图片以及对应的预测结果
    """
    # 去除批量维度，转换为 28x28 图片
    img_array = img_tensor.squeeze(0).squeeze(0).numpy()
    # 显示图片
    plt.imshow(img_array, cmap='gray')
    plt.title(f"Predicted Class: {labels[predicted_class]} (Index: {predicted_class})")
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 进行预测
predicted_class = predict_image(net, img_tensor)
# 定义标签
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 显示预测结果和图片
display_prediction(img_tensor, predicted_class, labels)