import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
''' 这是一个容器，用于将多个网络层按照顺序组合起来。它使得网络的定义更加简洁和有组织。
# 通常用于将卷积层的输出转换为全连接层可以处理的形状。 例如，假设输入是一个形状为 (batch_size, 1, 28, 28)
# 的图像（如 MNIST 手写数字数据集），nn.Flatten() 会将其转换为 (batch_size, 784)。
  
  一个全连接层（线性层），它接受大小为 784 的输入，并输出大小为 10 的结果
'''

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

'''
apply 是 nn.Module 的一个方法，用于递归地将给定的函数应用到模型的每一个子模块。
这里，它会遍历 net 中的所有层（在这个例子中是 nn.Flatten 和 nn.Linear），并对每个层调用 init_weights 函数。
'''

net.apply(init_weights);


loss = nn.CrossEntropyLoss(reduction='none')

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# 保存模型参数到文件
torch.save(net.state_dict(), '3.7model.pth')
print("Model saved as '3.7model.pth'")

# 加载模型
# 定义模型结构（必须和原始模型一致）
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# 加载保存的参数
net.load_state_dict(torch.load('3.7model.pth', weights_only=True))
print("Model loaded from '3.7model.pth'")

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)   针对报错，这个处理是 我装作看不见

''' 这里借助下书里面的函数 '''
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签"""
    for X, y in test_iter:    #就迭代一次 取出来最开始的256
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    '''  axis=1   按行找到每个行最大标签，
        output = torch.tensor([
                       [0.1,                         2.3,             1.8, 0.5, 0.7, 1.0, 2.1, 0.3, 1.5, 0.8],
                       [1.2, 0.3, 0.7,               2.5,             1.8, 0.2, 0.5, 0.1, 0.9, 0.4],
                       [0.5, 1.0, 0.3, 0.8, 1.5,     2.7,             0.6, 1.9, 0.2, 0.4]])
                       
        tensor([1, 3, 5])
    
    '''
    titles = ["true:   "+true +'\n' +"pred :  "+  pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 2, 3, titles=titles[0:n])
    '''                 1, n,  1行显示出来  '''

predict_ch3(net, test_iter)



'''另一种预测方式
# 从测试集中获取一个真实样本
for X, y in test_iter:
    real_example = X[0].unsqueeze(0)  # 取出一个样本并添加批量维度
    true_label = y[0]  # 真实标签
    break

# 使用模型预测
output = net(real_example)
predicted_class = torch.argmax(output, dim=1)

print("True label:", true_label.item())  # 打印真实标签
print("Predicted class:", predicted_class.item())  # 打印预测类别

但是没有图片

'''