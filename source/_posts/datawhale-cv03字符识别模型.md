---
title: datawhale-cv训练营-03字符识别模型
date: 2020-05-26 01:24:13
tags: [ML, CV]
---

前面的章节，我们讲解了赛题的背景知识和赛题数据的读取。 本文的任务是: 基于对赛题理解本章将构建一个定长多字符分类模型, 构建一个**CNN类**的定长字符识别模型。



## 3 字符识别模型

本章将会讲解卷积神经网络（Convolutional Neural Network, CNN）的常见层，并从头搭建一个字符识别模型。

### 3.1 学习目标

- 学习CNN基础和原理
- 使用Pytorch框架构建CNN模型，并完成训练

### 3.2 CNN介绍

卷积神经网络（简称CNN）是一类特殊的人工神经网络，是深度学习中重要的一个分支。

#### 为什么CV方面用CNN？

CNN在很多领域都表现优异，精度和速度比传统计算学习算法高很多。**特别是在计算机视觉领域**，CNN是解决图像分类、图像检索、物体检测和语义分割的主流模型。

#### 常见结构组成

CNN每一层由众多的卷积核组成，每个卷积核对输入的像素进行**卷积操作**，得到下一次的输入。随着网络层的增加卷积核会逐渐扩大感受野，并缩减图像的尺寸。

CNN是一种<u>**层次**</u>模型: 

##### 1. Input

输入的是原始的**像素数据**。



##### 2. 中间处理计算

CNN通过**卷积（convolution）、池化（pooling）、<u>非线性</u>激活函数（non-linear activation function）和全连接层（fully connected layer）**构成。

![卷积过程](https://github.com/datawhalechina/team-learning/raw/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/IMG/Task03/%E5%8D%B7%E7%A7%AF.png)

##### 3. Output

通过多次卷积和池化，CNN的**最后一层将输入的图像像素映射为具体的输出**：如在分类任务中会转换为不同类别的**概率输出**，



##### 4. 学习过程(优化参数)——反向传播



然后计算真实标签与CNN模型的输出的预测结果的**差异**，并通过**反向传播更新每层的参数**；在更新完成**后再次前向传播**，如此反复直到训练完成 。



![经典的字符识别模型](https://github.com/datawhalechina/team-learning/raw/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/IMG/Task03/Le_CNN.png)

### 3.3 模型的优势特点

传统机器学习相比，CNN, 或者说<font color="#dd0000">**深度学习模型**(各种**<u>深度神经网络</u>**)</font>，具有一种**<font color="#dd0000">端到端（End to End）的优势</font>**：模型训练的过程中是**<font color="#dd0000">直接</font>**从<u>输入</u>图像像素到<u>最终的输出</u>分类结果——<font color="#dd0000">并**不涉及**到具体的**特征提取**和构建模型的过程</font>，也不需要**人工的**参与。



### 3.4 实战：Pytorch构建CNN模型

在Pytorch中构建CNN模型非常简单，**只需要定义**好模型的**参数**和**正向传播函数**即可，<font color="#dd0000">Pytorch会根据正向传播**自动计算反向传播**</font>！！



#### 1. 定义模型

```Python
# 定义模型: 这个CNN模型包括两个卷积层，最后并联6个全连接层进行分类
class SVHN_Model1(nn.Module):
  	
    # 构造器：1. 只需要定义好模型参数
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),  
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(), 
            nn.MaxPool2d(2),
        )
        # 
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)
    
		# 2. 只需要定义好模型正向传播即可，Pytorch会根据正向传播自动计算反向传播。
    def forward(self, img):        
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6
    
# 构造一个模型对象
model = SVHN_Model1()
```



#### 2. 使用模型

在训练之前，需要定义好

```python
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), 0.005)
```



然后就是训练： 过程包括，Pytorch**自动计算**反向传播，让模型真正的实现<font color="#dd0000">“**自我学习**”</font>

```python
loss_plot, c0_plot = [], []
# 迭代10个Epoch
for epoch in range(10):
    for data in train_loader:
        c0, c1, c2, c3, c4, c5 = model(data[0])
        loss = criterion(c0, data[1][:, 0]) + \
                criterion(c1, data[1][:, 1]) + \
                criterion(c2, data[1][:, 2]) + \
                criterion(c3, data[1][:, 3]) + \
                criterion(c4, data[1][:, 4]) + \
                criterion(c5, data[1][:, 5])
        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_plot.append(loss.item())
        c0_plot.append((c0.argmax(1) == data[1][:, 0]).sum().item()*1.0 / c0.shape[0])
        
    print(epoch)
```





