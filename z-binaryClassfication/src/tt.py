# import cv2
# from matplotlib import pyplot as plt

# # 读取图片
# image_path = 'path_to_your_image.jpg'  # 替换为你的图片路径
# image = cv2.imread(image_path)

# # 检查图片是否成功读取
# if image is None:
#     print("Error: 图片读取失败")
# else:
#     # 转换颜色格式 BGR 到 RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # 显示图片
#     plt.imshow(image_rgb)
#     plt.axis('off')  # 不显示坐标轴
#     plt.show()

import os
import pandas as pd
currentpath = os.path.dirname(os.path.abspath(__file__))

currentpath = os.path.dirname(currentpath)

cueernpath = os.path



df = pd.read_csv("data/MRI/train/train.csv", sep=",", na_filter=False)


print(df.head())