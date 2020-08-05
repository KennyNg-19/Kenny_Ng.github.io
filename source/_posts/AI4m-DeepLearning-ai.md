---
title: DeepLearning.ai5月新课-AI4Medicince笔记
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

但这会和目的「让模型往更准确预测**不正常**图片的方向发展」相悖：the algorithm is optimizing  its updates to **get the normal examples,**  and not giving much relative weight to the mass examples. If learning with a highly unbalanced dataset, as we are seeing here, then the algorithm will be **incentivized(刺激) to prioritize the <u>majority class</u>**, since it contributes more to the loss.

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



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4dt34bvlj30vo09wgnp.jpg" alt="权重w，按各自病的数据分开算哦！" style="zoom:50%;" />



### c. Dataset Size问题：数据量

CNN，尤其是**规模更大的**CNN，是需要大量的labeled data做训练才会有比较好的效果。通常在医学影像的领域当中，在处理影像时只会包含10,000到100,000个不同的样本，像这样的资料量相对在其他领域例如分辨猫和狗的影像来说是非常少的，就是工程师们可能会遇到的Data Size的问题。

#### 2种解决的方式

##### **1. Transfer Learning，使用pre-training。**

下面这张图在pretraining的部分，这一个模型先被拿去做分辨企鹅的训练等到分辨企鹅的训练完成之后再转移到第二个步骤Fine-tuning，这时候工程师可以视情况决定需不需要升级这个训练模型，最后的目标是要让这一个模型可以分辨胸腔X光并且输出是哪一个疾病。

为什么用企鹅当作例子？

讲师在这边有提到因为企鹅整体的形状很像肺形状，所以在初期几层的神经网路当中可以去让模型分辨一些比较大的，例如边缘形状等等的问题，在第二次Fine -tuning的时候这个模型就会被修改并且能够分辨比较细节的部分。其实像这样的方式就是所谓的Transfer Learning。



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4ezmo6yvj30x80ey0y2.jpg" style="zoom:50%;" />

###### **题外话**：而当**数据充足**时，你可以：

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4h5xu7cmj31i40u0gst.jpg" alt="从头开始:新训练模型" style="zoom:33%;" />

##### **2. Data Augmentation数据增强**

如果喂给机器训练的影像数量太小的话，还有一种方法可以让影像变多！那就是以人工的方式将影像进行一些变化后再重新把变化过后的影像喂回机器做训练，这个就是Data Augmentation。

但是这边要注意一点像是胸腔X光来说，我们可以旋转图片重新喂给模型或是改变对比度去产生一张新的影像，然而如果我们将胸腔X光做**180度的翻转**时，这样机器就无法分辨他是正常的胸腔或是dextrocardia（右位心）——也就是说这种情况下，**label的状态改变**了，由正常的影像变成右位心的影像。

所以结论就是，当我们想要利用这种方式去增加影像的数量时，我们必须**考虑到这一张影像的label状态会不会改变**。

<img src="https://miro.medium.com/max/1390/1*PstRJ3s2Nw3mQEG3c5joug.png" alt="数据增强不是任意的，是有规范的" style="zoom:43%;" />

------



## II. 再聊训练验证和测试集的<u>进阶</u>问题

<img src="https://miro.medium.com/max/2279/1*GBsxeGzhvdr-iVTGc-dZaw.png" alt="训练、验证、测试的别名" style="zoom:50%;" />

一般来说机器学习会将我们所拥有的数据分做三组
1.Training set训练集 2.Validation set验证集 3.Test set测试集

分别有**别名**:

development set, tuning set 和 holdout set



## Dataset可能存在的进阶问题 

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4gt6lqzwj319g0bwtc2.jpg" style="zoom:50%;" />





### Q1: Data leakage 数据(因果关系)的泄露

#### 定义

Data Leakage 跟其他场合说的**数据安全数据泄漏**, **完全不一样**: 不是数据量因为泄露少了，**而是因果关系的提前泄漏**。是由于**前期**的**准备数据/数据采样**的时候出了问题，误将与**结果直接相关或存在颠倒因果关系的feature纳入了数据集**，使模型使用**颠倒的因果关系的数据**进行预测，得到**over-optimistic 过于乐观的**结果。



##### 具体实例: Patient <u>Overlap</u>

如果**重复的病人数据**被分别使用在训练组以及测试组的时候，机器可能会**”记忆“**该位病人的某项特殊特征，误将将这项特殊的特征当作是一个可以判断的依据，这种现象是Over-optimistic test set performance。就有点像是考试前你已经看过考题了一样，机器就会像这样子把答案记下来，并非像是你想让他做的-倚靠其他更有依据的线索找到答案。

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh6rdqztzuj310u0ng4e2.jpg" style="zoom:33%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh6r9x0iyej315g0lw7ca.jpg" alt="不同时期normal的照片加入2个set..." style="zoom:50%;" />



例如这张图片所示，是**normal的**病人在不同时期的照片，同时2次都带着项链。机器在test set中将这位病人的影像判定为正常，**可能**是**依据病人穿戴的项链**，而不是依据病人肺部的现象——不要小看DL model的记忆力: it's possible that the model actually **memorized to output normal when it saw the patient with a necklace on**. This is **not hypothetical**, deep learning models can unintentionally memorize training data, and the model could memorize **rare or unique training data aspects** of the patient,



#### 解决

解决方式其实很简单：只要**将同一位病人的数据放在<u>同一个组别</u>**即可——这样也是保证学习的目的：通过学习一些病人的特征，可以**泛化到**<u>更多的病人</u>身上。

以往在分数据的时候可能会从按照图像类别直接分——**保证分布的一致性**；但是医学影像的话反其道而行之，以病人作为拆分到不同组的根据。

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh6rkq9a50j30vh0u0wy5.jpg" alt="split by image VS patients" style="zoom:40%;" />





### Q2: Set sampling当数据本身分布不均匀时

#### 问题

就举医学影像为例，正常不患病的数量会远大于有病的医学影像数量，即分别问大类和小类。所以在分数据到不同的三个不同的数据集时，很有可能**测试组里面没有分到一张患病的影像**。



#### 解(同时引出此情况下，<u>各个集的sampling顺序</u>)

解决的方式是

1. 在分数据的时候被**设定至少有百分之X的**小类X会**被设成50%**

2. 在测试组的数据**确认之后**，接下来**要设定的就是验证组的数据**，验证集设定策略**和测试集合的基本上一样**

3. 当这两组的数据设定完之后**剩下的所有数据**，会被用作是**训练集**。

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4gx81troj31d20d87bu.jpg" style="zoom:33%;" />



##### 即data sampling顺序: 是test→validation→train

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh4gygxttdj30ba06m0tz.jpg" style="zoom:50%;" />





### Q3: Ground Truth("正确"的label)

#### 问题

在医学里面数据label，也就是学习结果的「正确答案」，在机器学习里面常**被称作Ground truth**， 而在医学上面同样的东西会被称作Reference Standard。

医学里面会常常**会有没有正确解答的现象**，就举胸腔X光来说，也许某一位放射科医师认为某张影像是肺炎，但同样的影像另外一位放射科医生可能**会有不同的意见**，这个叫做Inter-observer disagreement。



#### 定义"正确"的方法

如此一来决定Ground truth的方法也变得很重要，常见的方法有：

**1.Consensus Voting** ✅
就以胸腔X光来说，这个方式就是由一组放射科医生可能是<u>投票决定又或者是经由讨论达到某个共识</u>而决定最后的答案。

**2.Additional Medical Testing**

例如就像刚刚举例的胸腔X光，如果当放射科医生无法从胸腔X光得到最后的Ground truth时，这时病人会被建议去做其他的测试，例如CT，得到更精确的解答，验证胸腔X光的Ground truth。除了X光影像之外，例如皮肤癌照片也通常会由组织切片的验证结果才得到该照片的Ground truth。不过这个方法比较费时费力，所以**目前研究大多都是用第一个方式**。



------



## III. 再聊<u>分类问题</u>的evaluation metrics

### accuracy准确性的计算

#### accuracy定义和概率计算，有数学上的一致性

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghbb8xauogj319s0jytgv.jpg" alt="概率计算得到acc" style="zoom:43%;" />



所以只**要有sensitivity和specificity**，结合统计得到的prevalence，就可以**算出accuracy**



### 混淆矩阵

![Confusion Matrix](https://tva1.sinaimg.cn/large/007S8ZIlgy1gh6yd5wykaj31ho0kyn70.jpg)



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghb6apedduj30rw07wq4i.jpg" alt="注意这两个：实际是!" style="zoom:50%;" />



#### 助记！

##### 铺垫：什么NP R，TP R这种rate怎么算：

rate就不只是直接拿confusion matrix里的值了，而是**<font color="#dd0000">再除以对应行/列的<u>总和</u></font>**，即做分母!!

(以下均为rate)：

- 灵敏度(Sensitivity, **TPR**)和特异度(Specificity, **TNR**)，作为分母的是“**实际**有病/没病的总数“，所以可以看到是(TP，FN)和(TN, FP)这种<font color="#dd0000">**两两相反，四个字母完全不同**</font>的搭配！！

- 查准率PPV(Precision)和NPV，作为分母的是“**预测**有病/没病的总数“，所以可以看到是(TP，FP)和(TN, FN)这种<font color="#dd0000">**两两预测的pos和neg类相同，而结果的正确性true/false不同**</font>的搭配！！

  

- 查全率(Recall, Sensitivty) 和查准率(Precision, PPV)<font color="#dd0000">都是关于**postive类**的指标——**找到positve类(比如患病)**一般才是**ML预测的主要对象, 而不是关注negative类**</font>

------



### 先验(prevalence)做分母：sensitivity灵敏度(即查全率)和specificity特异度

> Sensitivity **only** considers output on people **in the positive class**
>
> Similarly, specificity **only** considers output on people **in the negative class**.
>

- **灵敏度/查全率(recall) P(+|disease)**：model预测为pos且实际为pos/所有实际为pos(TP+**FN**)，的比例

  理解<u>**查全**</u>：**在实际pos的样本中，model预测为正且预测正确**的比例——**非漏诊性！越高，说明放过<u>更少的</u>患病者**

  计算公式为：TPR=TP/ (TP+ FN)



- **特异度P(—|non-disease)，P(—|neg)** 简称**TNR，TN <u>Rate</u>**：model预测为neg且实际为neg/所有实际为neg，的比例——**分母是<font color="#dd0000">non-disease，诊断没病的查全率</font>，非误诊性！越高，说明<u>将越多的非病人正确排除</u>**

  计算公式为：TNR= TN / (FP + TN)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghba7wvrgaj30xf0u04qp.jpg" alt="灵敏度和特异度的解释" style="zoom:43%;" />

##### 结论：敏感度高=漏诊率低，查全率高；特异度高=误诊率低



注：(P(+|disease)<u>分子的+</u>,是**model**预测的结果！！！)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh6v7xrtswj314w0hswkh.jpg" alt="注意: 先验概率+是model判定为正的！！" style="zoom:33%;" />



##### 练习题：坑

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghbd7v75vtj317y0tw7cp.jpg" style="zoom:40%;" />



### ROC曲线

#### 目的

理想情况下我们**希望敏感度和特异度都很高**，然而实际上一般在敏感度和特异度中**寻找一个平衡点**，这个过程可以**用ROC(Receiver Operating Characteristic)曲线**来表示, 和 [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)值(Area Under the Curve) 来精确表示：



#### 定义

- 纵坐标：**RPR，sensitivity**
- 横坐标：**FPR = 1 — specificity(TNR)**——**误诊率**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghbg9nfon6j30f80fi42l.jpg" style="zoom:50%;" />

#### 为什么横坐标用误诊率？1 - 特异性？

为了满足：**Sensitivity、Specificity这两个指标越大**的情况

理想情况下，**TPR应该接近1，FPR应该接近0**, 即TPR=1，FPR=0，即**图中(0,1)点。故ROC曲线越靠拢(0,1)点，越偏离45度对角线越好**，**Sensitivity、Specificity越大**效果越好。



#### 深度理解ROC

ROC曲线的**横坐标和纵坐标其实是<font color="#dd0000">没有相关性的</font>**，所以<font color="#dd0000">**不能把ROC曲线当做一个函数曲线来分析，应该把ROC曲线看成无数个点**</font>，每个点都代表一个分类器，其横纵坐标表征了<font color="#dd0000">这个**分类器的性能**</font>。为了更好的理解ROC曲线，我们先引入ROC空间



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghbfjctbu0j30n80nawhi.jpg" alt="ROC space" style="zoom:50%;" />

明显的，C'的性能最好。而B的准确率只有0.5，几乎是随机分类。特别的，图中左上角坐标为（1,0）的点为**完美分类点（perfect classification），它代表所有的分类全部正确**，即模型**预测为1的点全部正确（TPR=1），归为0的点没有错误（FPR=0）**。



通过ROC空间，我们明白了一条ROC曲线其实**代表了无数个分类器**：那么我们为什么常常用一条ROC曲线来**描述一个分类器**呢？

仔细观察ROC曲线，发现其都是上升的曲线（斜率大于0），且都通过点（0,0）和点（1,1）。其实，这些点是一个个的分类器，而每个分类器实际**习得的<font color="#dd0000">也是一个最佳阈值</font>**。所以ROC,可以代表着**一个**分类器**在不同阈值下**的分类效果，即曲线从左往右可以**认为是阈值**变化过程——但是**<font color="#dd0000">不一定是从0到1的，也可能是反过来——得看具体场景</font>**



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghbcutbqclj30z50u0awm.jpg" alt="完美划分时, 无论阈值是什么总恰有一个坐标=1" style="zoom:43%;" />



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghbczdv3vej30es0fawgl.jpg" alt="完美划分时的ROC" style="zoom:33%;" />





#### **AUC**值

那么AUC值的含义是什么呢？

> The AUC value is equivalent to the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

这句话有些绕，我尝试解释一下：首先**AUC值是一个<u>概率值</u>**——随机挑选一个正样本以及一个负样本，当前的分类算法，根据计算得到的Score值并**将这个正样本排在负样本前面的概率**就是AUC值。当然，**AUC值越大**，当前的分类算法**越有可能**将正样本排在负样本前面，即能够更好的分类。



从AUC判断分类器（预测模型）优劣的标准：

- AUC = 1，是完美分类器，采用这个预测模型时，**<font color="#dd0000">存在至少一个</font>阈值能得出完美预测**(绝大多数预测的场合，不存在完美分类器)
- 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
- AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。
- AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。

三种AUC值示例：

![img](https://pic3.zhimg.com/80/v2-f9d1bf42ddcaaab151464e1d2e9f1d30_1440w.jpg)

简单说：**AUC值越大的分类器，正确率越高**



#### 为什么使用ROC曲线

既然已经这么多评价标准，为什么还要使用ROC和AUC呢？

因为ROC曲线有个**很好的特性**：当测试集中的**正负样本的分布变化的时候，ROC曲线能够保持不变**——在实际的数据集中经常会出现类**不平衡（class imbalance）**现象，即负样本比正样本多很多（或者相反），而且测试数据中的正负样本的分布也可能随着时间变化

------



### 后验(预测结果)做分母：PPV查准率(又Precision)、NPV

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh6vcxavpoj30v80awq6f.jpg" alt="PPV" style="zoom:33%;" />



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh6vdcaqw1j30vq0amdjl.jpg" alt="NPV" style="zoom:33%;" />

------



### PRC曲线(precision-recall)

#### 目的

和ROC曲线用于权衡灵敏度和特异度的作用类似，理想情况下我们**希望precision和recall都高"**， "实际上一般在敏感度和特异度中**寻找一个平衡点**，

ROC shows the trade-off between precision and recall for different thresholds. A high area under the curve represents both high recall and high precision,



#### F1-score(量化PRC)

同理，类似AUC值的精确作用，F1 score: harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.

# 2. Course2-AI for Medical Prognosis(预断,预后)

