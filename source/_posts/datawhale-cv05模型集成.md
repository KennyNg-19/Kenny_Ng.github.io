---
title: datawhale-cv05模型集成
date: 2020-06-02 22:35:07
tags: [ML, CV]
---



本章是本次赛题学习的最后一章，将会讲解如何使用集成学习提高预测精度



## 学习目标

本章讲解的知识点包括

- 学习集成学习**方法**，以及交叉验证情况下的模型集成
- 学会使用**深度学习**模型的集成学习和结果后处理思路。

## 5 模型集成

### 5.1 集成学习方法

在机器学习中的集成学习可以在一定程度上提高预测精度，常见的集成学习方法有Stacking、Bagging和Boosting，同时这些集成学习方法与具体验证集划分联系紧密。

由于深度学习模型一般需要较长的训练周期，如果硬件设备不允许建议选取留出法，如果需要追求精度可以使用交叉验证的方法。

下面假设构建了10折交叉验证，训练得到10个CNN模型。



![集成学习](https://github.com/datawhalechina/team-learning/raw/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/IMG/Task05/%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81.png)



那么在10个CNN模型可以使用如下方式进行集成：

1. 对预测的结果的<u>概率值</u>**进行平均**，然后解码为具体字符；

2. or 对预测的字符进行**投票**，得到最终字符



#### 缺点

训练了**10个CNN**，**训练时间**较长



### 5.2 深度学习中的集成学习

此外在深度学习中本身还有一些集成学习思路的做法，值得借鉴学习：

#### 5.2.1 Dropout

Dropout可以作为<font color="#dd0000">**训练**</font>深度神经网络的一种技巧。在每个<font color="#dd0000">**训练**批次</font>中，通过<font color="#dd0000">**随机让一部分(按照输入的比例)的节点**停止工作</font>，但同时在<font color="#dd0000">**预测**的过程中</font>让<font color="#dd0000">**所有的**节点</font>都其作用。

##### 作用

经常出现在在**先有的CNN网络**中，可以有效的**缓解模型过拟合**的情况，也可以在预测时**增加模型的精度**



##### 代码

加入Dropout后的代码如下: 即nn**.Dropout(0.25)**这一行，(按照输入的比例为0.25)

```Python
# 定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.25), //
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(), 
            nn.Dropout(0.25), //
            nn.MaxPool2d(2),
        )
        # 
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)
    
    def forward(self, img):    
      //...
```



#### 5.2.2 Snapshot

本章的开头5.2已经提到训练了**10个CNN**则可以将多个模型的预测结果**进行平均**。但是加入只训练了**<u>一个</u>**CNN模型，如何做模型集成呢？

在论文Snapshot Ensembles中，作者提出使用cyclical learning rate进行训练模型，并<font color="#dd0000">保存**精度比较好的**一些checkopint，最后**将多个checkpoint**进行**模型集成**</font>。


![Snapshot](https://github.com/datawhalechina/team-learning/raw/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/IMG/Task05/Snapshot.png)

##### 优/缺点

在Snapshot论文中作者通过使用表明，此种方法

- 优点：可以在一定程度上**提高模型精度**，

- 缺点：但**需要更长的训练时间**；由于在cyclical learning rate中**学习率**有周期性变大和减少的行为，因此CNN模型很有可能在**跳出局部最优进入<u>另一个局部最优</u>**。



#### 5.2.3 TTA<font color="#dd0000">**<u>测试集</u>**</font>数据扩增

<font color="#dd0000">**<u>测试集</u>**</font>数据扩增（Test Time Augmentation，简称TTA）也是常用的集成学习技巧。



数据扩增<font color="#dd0000">不仅可以在训练时候用，而且可以同样在预测时候进行数据扩增</font>，对同一个样本预测三次，然后对三次结果进行平均。

| 1                                                            | 2                                                            | 3                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![IMG](https://github.com/datawhalechina/team-learning/raw/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/IMG/Task02/23.png) | ![IMG](https://github.com/datawhalechina/team-learning/raw/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/IMG/Task02/23_1.png) | ![IMG](https://github.com/datawhalechina/team-learning/raw/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/IMG/Task02/23_2.png) |



##### 代码

```python
def predict(test_loader, model, tta=10):
   model.eval()
   test_pred_tta = None
  
   # TTA 次数(对测试集做数据扩增)
   for _ in range(tta): //
       test_pred = []
   
       with torch.no_grad():
           for i, (input, target) in enumerate(test_loader):
               c0, c1, c2, c3, c4, c5 = model(data[0])
               output = np.concatenate([c0.data.numpy(), c1.data.numpy(),
                  c2.data.numpy(), c3.data.numpy(),
                  c4.data.numpy(), c5.data.numpy()], axis=1)
               test_pred.append(output)
       
       test_pred = np.vstack(test_pred)
       if test_pred_tta is None:
           test_pred_tta = test_pred
       else:
           test_pred_tta += test_pred
   
   return test_pred_tta
```





## 6. 结果后的额外处理

在不同的任务中可能会有不同的解决方案，不同思路的模型**不仅可以互相借鉴**，同时也可以**修正**最终的预测结果。



在本次赛题中，可以从以下几个思路对预测结果进行后处理：

- 统计图片中每个位置字符出现的**频率**，使用**<u>规则</u>修正结果**；
- 单独训练一个**字符长度预测模型**，用来预测图片中**字符个数，并修正结果**