---
title: datawhale-cv训练营-01赛题研究
date: 2020-05-19 00:53:16
tags: [ML, CV]
---

这次**基础**赛事, 是Datawhale与天池联合发起的零基础**入门系列**赛事 [赛事地址](https://tianchi.aliyun.com/competition/entrance/531795/introduction)

# 本文目的

1. **总结**基本了解比赛规则
2. **总结**解题思路
3. 数据下载和**理解**

# 1.规则

本赛题需要选手识别图片中所有的字符。**评测指标**：准确率，Score=编码识别正确的数量/测试集图片数量

为了降低比赛难度，我们提供了训练集、验证集中所有字符的**位置框**（在**阿里天池**上下载）。

**注意**: 按照比赛规则，所有的参赛选手，**只能使用比赛给定的数据集完成训练(不要使用SVHN原始数据集进行训练**）

#### 使用的Python模块

大概介绍一下，这里可能需要用到的主要模块。

> numpy ：提供了python对多维数组对象的支持：ndarray，具有矢量运算能力，快速、节省空间。numpy支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
>
> torch：神经网络界的 Numpy, 因为他能将 torch 产生的 tensor 放在 GPU 中加速运算 (前提是你有合适的 GPU), 就像 Numpy 会把 array 放在 CPU 中加速运算. 所以神经网络的话, 当然是用 Torch 的 tensor 形式数据最好
>
> torchvision：torchvision包是服务于pytorch深度学习框架的,用来生成图片,视频数据集,和一些流行的模型类和预训练模型。我认为这个是最关键的模块
>
> OpenCV（import时候是cv2）：一款强大的跨平台的计算机视觉库，使用它能完成我们对于图像和视频处理的很多功能。它以电信号的方式加以捕捉、记录、处理、储存、传送与重现的各种技术。这里主要是用来对图片的处理
>
> json：这个就是json的读写库，处理json文件的

# 2. 数据理解

#### 数据集初步观察

分.json的label位置信息，和原图集合

**数据读取**： json文件包含的位置信息，除了便于正式的训练，还可以用于数据观察——直接作用在原图集，**查看已给的位置信息的分割效果**

样例代码: 数据读取，在此我们给出JSON中标签的读取方式

```
import json
train_json = json.load(open('../input/train.json'))

# 数据标注处理
def parse_json(d):
   arr = np.array([
       d['top'], d['height'], d['left'],  d['width'], d['label']
   ])
   arr = arr.astype(int)
   return arr

img = cv2.imread('../input/train/000000.png')
arr = parse_json(train_json['000000.png'])

plt.figure(figsize=(10, 10))
plt.subplot(1, arr.shape[1]+1, 1)
plt.imshow(img)
plt.xticks([]); plt.yticks([])

for idx in range(arr.shape[1]):
   plt.subplot(1, arr.shape[1]+1, idx+2)
   plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])
   plt.title(arr[4, idx])
   plt.xticks([]); plt.yticks([])
```

输出

[![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gez59nbvuij30z30u0gtm.jpg)](https://tva1.sinaimg.cn/large/007S8ZIlgy1gez59nbvuij30z30u0gtm.jpg)

[![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gez5ab60sij30yw0oaqc0.jpg)](https://tva1.sinaimg.cn/large/007S8ZIlgy1gez5ab60sij30yw0oaqc0.jpg)

# 3. 潜在的疑难杂症

#### 预处理细节

数据集存在原图片**大小不统一**，这个只需要用pytorch的transforms处理即可

#### 难点

##### a.确定待识别数字在图中的位置

(使用比赛简化后的数据，则该问题并不存在了)

在简化数据集之前的难点是：模型要能找到待识别数字的**位置**。但是既然处理后的数据集，**位置信息全都提供了**，那么这个问题就容易很多——**单纯的识别数字信息**。数字的位数问题可以通过简单的算法来解决，就像MNIST数据集一样。

[![图1](https://crazy-winds.github.io/images/cv1/0-1.png)](https://crazy-winds.github.io/images/cv1/0-1.png)图1

（json格式存储label和位置信息）

[![图2](https://crazy-winds.github.io/images/cv1/0-2.png)](https://crazy-winds.github.io/images/cv1/0-2.png)图2

##### b.确定待识别数字的个数

即每幅图的数字个数可能均不相同，如何统一的解决(找到一种general的方法)

将在解题思路部分详细展开



# 4. 大致解题思路

1. 简单入门思路：定长字符识别。将不定长字符转化为定长处理，不足部分用**填充占位符**为代替

（**适合新手**也**适合此题给的处理后的数据集**：赛题数据集中大部分图像中字符个数为2-4个，最多的字符 个数为**6个**）

[![图3](https://crazy-winds.github.io/images/cv1/0-3.png)](https://crazy-winds.github.io/images/cv1/0-3.png)图3

1. 专业字符识别思路：按照**不定长字符**处理

在字符识别研究中，有**特定的方法**来解决此种不定长的字符识别问题：如**典型的有CRNN字符识别模型**。

因为本次赛题中给定的图像数据都**比较规整，可以视为一个单词或者一个句子** 喂进CRNN模型。

1. 专业分类思路：检测位置再识别数字

在赛题数据中已经给出了训练集、验证集中所有图片中字符的位置，因此可以首先将字符的位置进行识别，利用**物体检测**的思路完成。

可参考物体检测模型：**SSD或者YOLO**