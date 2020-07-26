---
title: DeepLearning.ai的5月新课AI 4 medicince笔记
date: 2020-07-25 17:26:33
tags: [advancedML, ML, CV, math]
---



# 1. Course1-AI for Medical Diagnosis(诊断)

## I. ML进阶，可能遇到的**<u>更实际</u>的**<u>进阶</u>问题：3个

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4delyjewj31520f2dmd.jpg" style="zoom:50%;" />

### a. 训练数据分布-Class imbalance问题

如 实际在收集要拿來做DL的医学数据時，常识是 **正常的非病患的数据比不正常的、有疾病的多上很多**

loss function用的是: **Binary cross-entropy loss**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh3d7i1s7tj30x606cwg8.jpg" style="zoom:33%;" />

model输出是：P(不正常影像), 对于**不正常影像(y=1)的预测**！(不是看label预测哦，测试数据压根就没有label)



#### 导致问题：impact on **Loss calculation**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh3daki4igj31g60oo7ge.jpg" alt="假设初始的classifier预测P均为0.5" style="zoom:30%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh3dbxlo62j30rs07k0wq.jpg" alt="loss占比越大的，就是优化的bias" style="zoom:33%;" />

但这会和目的「让模型往更准确预测**不正常**图片的方向发展」相悖：the algorithm is optimizing  its updates to **get the normal examples,**  and not giving much relative weight to the mass examples

#### 解决：

##### 解1 给loss加上权重：weighted loss

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh3dk2fxymj30y20hgdky.jpg" alt="weight就是类数量比的反向" style="zoom:33%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh3djqzy7aj315a0ju480.jpg" alt="可以让loss的贡献接近" style="zoom:33%;" />

##### 解2  人工重新采样Resampling恢复balance(有缺点)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh3dvgvajhj31eq0mawt7.jpg" style="zoom:33%;" />

原本的training set有6張正常的影像，和2張不正常的影像，以**人工的方式**將正常的影像**去掉2張之後**+再把2張不正常的影像**重复**放入變成4張。

这样，正常的影像和不正常的影像都分別有四張，便可以解決先前的Class Imbalance 問題。但是這個解決方式會有明显的缺点：原本该有6種不同的正常影像，变成了4種——也就是說會有**部分数据 沒有参与這次训练**+而有(**重复数据参与了训练**)



### b. Multi-task问题: 让模型一次性做<u>多重的</u>分类任务

在一开始的例子当中，我们只以有肿块mass和没有砖块来做区分，但是现实生活中的医学影像却是不只有两种情况，就如胸腔Ｘ光来说，可能会发现有许多**其他种类**的疾病，例如：肿块、肺炎、肺积水…等。

当我们想要**<u>同时</u>**检测的疾病**种类不只一个**的时候这样就是**属于Multi-task的问题**

#### 解决: 只用一个模型。好处？

如何建立**一个**具有处理Multi-task(多重任务)问题的模型？只用一个模型的**好处**？can **learn features** that are **common** to identifying more than one disease,  allowing us to **use our existing data more efficiently**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4dkfbai0j31ao0ba7as.jpg" alt="1 model" style="zoom:33%;" />



#### Mult-task和单任务比，的不同

如图中所示加入我们要在X光胸腔分辨mass、pneumonia和Edema时, 可以看到左边所显示的是**每一张图片的desired label**+右边是机器给的prediction probability值，这时候计算多重任务的Loss时他只要像右上角说显示的一样**将各别的Loss值相加**就可以。这样所计算出来的加总过后的Loss就会是训练更新模型的Loss。

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4diz8x92j30xc0gawjg.jpg" alt="训练examples和loss形式的对应改变" style="zoom:43%;" />

data分布，也可能出class imbalance问题：像这样一次处理许多疾病，在收集资料的时候也有可能出现比重不均的情况——所以在多重任务的情况下也可以使用解决方法的加上权重。

![Image for post](https://miro.medium.com/max/60/1*pOQ5doanwmA6jgpHMUaE2w.png?q=20)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4dt34bvlj30vo09wgnp.jpg" alt="权重w，按各自病的数据分开算哦！" style="zoom:50%;" />



### c. Dataset Size问题：数据量

CNN，尤其是**规模更大的**CNN，是需要大量的labeled data做训练才会有比较好的效果。通常在医学影像的领域当中，在处理影像时只会包含10,000到100,000个不同的样本，像这样的资料量相对在其他领域例如分辨猫和狗的影像来说是非常少的，就是工程师们可能会遇到的Data Size的问题。

#### 2种解决的方式

##### **1. Transfer Learning，使用pre-training。**

下面这张图在pretraining的部分，这一个模型先被拿去做分辨企鹅的训练等到分辨企鹅的训练完成之后再转移到第二个步骤Fine-tuning，这时候工程师可以视情况决定需不需要升级这个训练模型，最后的目标是要让这一个模型可以分辨胸腔X光并且输出是哪一个疾病。

为什么用企鹅当作例子？

讲师在这边有提到因为企鹅整体的形状很像肺形状，所以在初期几层的神经网路当中可以去让模型分辨一些比较大的，例如边缘形状等等的问题，在第二次Fine -tuning的时候这个模型就会被修改并且能够分辨比较细节的部分。其实像这样的方式就是所谓的Transfer Learning。

![Image for post](https://miro.medium.com/max/60/1*eHrveqPrTYSDYJnE8Ai5wg.png?q=20)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4ezmo6yvj30x80ey0y2.jpg" style="zoom:50%;" />

###### **题外话**：而当**数据充足**时，你可以：

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4h5xu7cmj31i40u0gst.jpg" alt="从头开始:新训练模型" style="zoom:33%;" />

##### **2. Data Augmentation数据增强**

如果喂给机器训练的影像数量太小的话，还有一种方法可以让影像变多！那就是以人工的方式将影像进行一些变化后再重新把变化过后的影像喂回机器做训练，这个就是Data Augmentation。

但是这边要注意一点像是胸腔X光来说，我们可以旋转图片重新喂给模型或是改变对比度去产生一张新的影像，然而如果我们将胸腔X光做180度的翻转时，这样机器就无法分辨他是正常的胸腔或是dextrocardia（右位心）也就是说这种情况下，label的状态改变了，由正常的影像变成右位心的影像。所以结论就是，当我们想要利用这种方式去增加影像的数量时，我们必须考虑到这一张影像的label状态会不会改变。

<img src="https://miro.medium.com/max/1390/1*PstRJ3s2Nw3mQEG3c5joug.png" alt="数据增强是有要求的" style="zoom:43%;" />



## II. 再聊训练验证和测试集的<u>进阶</u>问题

<img src="https://miro.medium.com/max/2279/1*GBsxeGzhvdr-iVTGc-dZaw.png" alt="训练、验证、测试的别名" style="zoom:50%;" />

一般来说机器学习会将我们所拥有的数据分做三组
1.Training set训练集 2.Validation set验证集 3.Test set测试集

分别有**别名**:

development set, tuning set 和 holdout set



## Dataset可能存在的进阶问题 

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4gt6lqzwj319g0bwtc2.jpg" style="zoom:50%;" />





### Q1: Data leakage 数据(因果关系)的泄露

Data Leakage 跟其他场合说的**数据安全数据泄漏**, **完全不一样**: 不是数据量因为泄露少了，**而是因果关系的提前泄漏**。是由于**前期**的**准备数据/数据采样**的时候出了问题，误将与**结果直接相关或存在颠倒因果关系的feature纳入了数据集**，使模型使用**颠倒的因果关系的数据**进行预测，得到**over-optimistic 过于乐观的**结果。



具体例子：如果**重复的病人数据**被分别使用在训练组以及测试组的时候，机器可能会**”记忆“**该位病人的某项特殊特征，误将将这项特殊的特征当作是一个可以判断的依据，这种现象是Over-optimistic test set performance。就有点像是考试前你已经看过考题了一样，机器就会像这样子把答案记下来，并非像是你想让他做的-倚靠其他更有依据的线索找到答案。

![Image for post](https://miro.medium.com/max/66/1*DUFe5zj6lhaO1zBnEZ3ZMw.png?q=20)

![Image for post](https://miro.medium.com/max/2847/1*DUFe5zj6lhaO1zBnEZ3ZMw.png)



例如这张图片所示，机器将这位病人的影像判定为正常，是依据病人穿戴的项链，而不是依据病人肺部的现象。

> 职业病碎念：请大家照X光的时候一定要把项链拿掉！！

这样子的解决方式其实很简单只要将同一位病人的数据放在同一个组别即可，例如同一个病人的多张影像同时放在训练组或是测试组，所以以往在分数据的时候可能会从影像上直接拆分，但是医学影像的话必须以病人作为拆分数据到不同组的根据。



### Q2: Set sampling

#### 解

![image-20200726164601303](https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4gx81troj31d20d87bu.jpg)



##### 注意data sampling顺序: 是test-val-train

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4gygxttdj30ba06m0tz.jpg" alt="image-20200726164711350" style="zoom:50%;" />



### Q3: Ground Truth





# 2. Course2-AI for Medical Prognosis(预断,预后)

