---
title: ML,DL的混淆点/易错点
date: 2020-07-25 13:00:22
tags: [ML, math, 统计概率, 积累]
---

# ML(数学)中的常见混淆, 易错点和一般性结论



## 0. 似然？

似然和概率在统计学中是经常见到的两个术语，有时候这两个概念是一个意思，有时候却有很大区别。这里梳理下这两个术语所代表的具体含义。

#### 本文中数学符号及含义

| 符号                                                         | 含义             |
| ------------------------------------------------------------ | ---------------- |
| O                                                            | 观测值           |
| θ                                                            | 随机过程中的参数 |
| <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gifp45h7myj303c02q745.jpg" style="zoom:25%;" /> | 参数的估计       |
| P(O\|θ)                                                      | 概率             |
| L(θ\|O)                                                      | **似然函数**     |

#### wiki中关于“似然”和“概率”的解释

- 在频率推论中，似然函数（常常简称为似然）是一个在**给定了数据**后**关于模型参数的**函数。**在非正式情况下，“似然”通常被用作“概率”的同义词。**
- 在**<u>数理统计</u>**中，两个术语**<u>则有不同的意思</u>**：
  - “概率”描述了**给定模型参数后**，输出**结果的合理性(可能性大小)**，而不涉及任何观察到的数据。
  - “似然”则描述了**给定了特定观测结果**后，描述模型**参数是否合理**

#### 似然函数公式

先看似然函数的定义，它是给定联合样本值x下关于(未知)参数θ的函数：(等号应该是**<u>正比于</u>**)![](https://www.zhihu.com/equation?tex=L%28%5Ctheta+%7C+%5Ctextbf%7Bx%7D%29+%3D+f%28%5Ctextbf%7Bx%7D+%7C+%5Ctheta%29)

- 这里的小x 指联合样本随机变量X取到的值，即X=x；

- 这里的θ是指未知参数，属于参数空间；

- 这里的f(x|θ)是一个**概率密度函数**，特别地表示(给定)θ下，关于联合样本值x的**联合密度函数**。

所以从定义上，似然函数和密度函数是完全不同的两个**数学对象**：前者是关于θ的函数，后者是关x的函数。所以这里的等号=，应理解为**<font color="#dd0000">函数值形式</font>的相等**，而<font color="#dd0000">不是两个函数本身是同一函数</font>。所以这个式子的**严格书写方式**是![](https://www.zhihu.com/equation?tex=L%28%5Ctheta+%7C+%5Ctextbf%7Bx%7D%29+%3D+f%28%5Ctextbf%7Bx%7D+%3B+%5Ctheta%29)(分号示把参数隔开) 即**θ在右端只当作参数**。



#### 注意：计算题中，贝叶斯具体求后验时，似然值怎么找——已经设为定值的分母, 则与分母无关！

![若有分母一定时，直觉上可得的分子概率，那就是似然值！](https://tva1.sinaimg.cn/large/007S8ZIlgy1gji415ep10j30fa03udhm.jpg)



#### 概率角度看ML: 频率派 vs 贝叶斯派

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjo3vb3pqgj31450u016u.jpg" alt="概率角度分析两派的机器学习" style="zoom:50%;" />

**频率派，统计机器学习: MLE 极大后验估计**
<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjo3zx4hv4j30e20agwie.jpg" alt="MLE 极大后验估计" style="zoom:50%;"/>

**贝叶斯派机器学习**：**MAP，极大后验估计**，**对概率公式积分——求概率值**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjo41bewn3j30to0lw4ef.jpg" alt="MAP 考虑了prior的极大似然估计" style="zoom:40%;" />





#### 对比本质: “似然”和“概率”是站在<u>两个角度</u>看待问题

##### 例如

- 概率

> 抛一枚均匀的硬币，拋20次，问15次拋得正面的可能性有多大？ 这里的可能性就是”概率”，均匀的硬币就是给定参数θ=0.5，“拋20次15次正面”是观测值O。求概率P(H=15|θ=0.5)=？的概率。

- **“似然”描述了给定了特定观测值后，描述<u>模型参数</u>是否合理。**

> 拋一枚硬币，拋20次，结果15次正面向上，问其为均匀的可能性？ 这里的可能性就是”似然”
>
> “拋20次15次正面”为观测值O为已知，参数θ并不知道，求L(θ|H=15)=P(H=15|θ=0.5)(prior和evidence都是常数) 的**最大化下的θ 值**

例B. 对于统计**概率分布**的似然



**最大似然估计**：given data, 寻找**概率最大的概率分布**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjie9ys7s0j31qk0tekhj.jpg" alt="最大似然估计" style="zoom:30%;" />



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjie8c3kjpj315m0bedi7.jpg" alt="given data, 在寻找最优分布" style="zoom:33%;" />





#### 离散随机变量的“似然”与“概率”



#### 连续型随机变量的“似然”与“概率”



## 1. 均值化？归一化？傻傻分不清

"标准化"和"归一化"这两个中文词要指代**四种**Feature scaling(特征缩放)方法, 实质是<font color="#dd0000">**一种线性变换**: **对向量 X 按照比例α压缩, 再进行平移**</font>。线性变换有很多良好的性质，这些性质**决定了对数据改变后<u>不会造成“失效”，反而能提高数据的表现</u>**，这些性质是归一化/标准化的前提, 详情见[特征工程中的「归一化」有什么作用？- 知乎](https://www.zhihu.com/question/20455227/answer/370658612) 。



先说**<u>数学领域</u>的normalization <u>标准化</u>**: 泛指将一组数，**<u>除以它们的sum</u>**, **得到各自占比且总和为1**——常见的就是求**概率值**或**概率分布**



再到了**<u>ML方面处理feature</u>**时，具体这四种常用的分别是：

1.  **归一化** Rescaling (**min-max normalization**)  ，![[归一化]](https://www.zhihu.com/equation?tex=x%5E%7B%27%7D+%3D+%5Cfrac%7Bx-min%28x%29%7D%7Bmax%28x%29-min%28x%29%7D) 

2. **均值归一化** mean normalization  ![[均值归一化]](https://www.zhihu.com/equation?tex=x%5E%7B%27%7D+%3D+%5Cfrac%7Bx-mean%28x%29%7D%7Bmax%28x%29-min%28x%29%7D+) 

   先说**x - mean**：这叫**centralize中心化**。因为 sum = mean * N, 而处理是每一项减去mean就是总共减去了mean * N，这样可以保证**处理后的sum为0**！

   

   归一化性质总结：把数据变成(0,1)或者（1,1）之间的小数。主要是为了数据处理方便提出来的，把数据映射到0～1**范围**之内处理，更加便捷快速

   

   目的关键词：归一化的缩放，顾名思义，"归一"——**“拍扁”压缩到<font color="#dd0000">区间</font>（仅由极值决定）**

3.  **标准化 Standardization**(**<u>Z-score</u> normalization**)  ![[标准化]](https://www.zhihu.com/equation?tex=+x%5E%7B%27%7D+%3D+%5Cfrac%7Bx-mean%28x%29%7D%7B%5Csigma%7D) 

   标准化总结：**能伸能缩**，当数据较为集中时， α更小，于是数据在标准化后就会**更加分散**。如果数据本身分布很广，那么 α 较大，数据就会被**更加集中**，到更小的范围内。

   目的关键词：标准化的缩放**是更加“弹性”和“动态”的能伸能缩，和<font color="#dd0000">整体样本的分布</font>有很大的关系，每个点都在贡献缩放，通过方差（variance）体现出来**。

   补充说明：

   - 在Batch Normalization(BN)出现之前是很有必要的，因为这样**拉到均值附近，学习的时候更加容易**，毕竟**激活函数是以均值为中心的，学习到这个位置才能将不同的类分开**。

   -  但是BN出现之后，这个操作就完全没必要了。因为每次卷积后都有BN操作，BN就是把数据拉到0均值1方差的分布，而且这个均值和方差是动态统计的，不是只在原始输入上统计，因此更加准确。

4. 中心化：x' = x - μ

   平均值为0，对标准差无要求

5.  Scaling to unit length  ![](https://www.zhihu.com/equation?tex=x%5E%7B%27%7D+%3D+%5Cfrac%7Bx%7D%7B%7C%7Cx%7C%7C%7D) 



什么时候用归一化？什么时候用标准化？

- 归一化： 输出范围在0-1之间
- 标准化：输出范围是**负无穷到正无穷**，灵活

  （1）如果对输出结果**<u>范围</u>有要求**，用归一化。
            如果数据较为稳定，**不存在极端的最大最小值**，用归一化。
  （2）如果数据存在**异常值和较多噪音**，用**标准化**，可以间接通过中心化避免异常值和极端值的影响。

**一般来说，我个人建议<u><font color="#dd0000">优先使用标准化</font></u>。**在对特征形式有要求时再尝试别的方法，如归一化或者更加复杂的方法。很多方法都可以将输出调整到0-1，如果我们对于数据的分布有假设的话，更加有效方法是使用相对应的概率密度函数来转换



## 2. 理清ML flow各个部分的关系

从上到下是 应用层→数据底层：

1. 最优模型(hypothesis)，作为ML flow的最红产物，本质是为**实际业务**服务的——最终用于输入实际数据，输出需要的结果

2. cost/error func是为**寻找最优模型**服务的，作为metric, 找出**最优参数**——是最优化算法去**优化**的对象，(输出是错误metric, 仅用于评估), 但实际为模型服务的是是取到极值的最优参数——正则化项也在其中(既然是为了调参)

3. 学习/优化算法，是为**cost func**服务的——**通过数学方法让metric快速取到极值**



## 3. 训练的目的是输出最佳参数

用训练数据的训练过程，根本目的是**输出最佳的参数(即带着该参数的hypothesis)，而非<u>最低的错误率——错误率只是判定的指标</u>**。所以相应的**简化训练过程/提高效率的技巧**，如PCA降维，只能用在训练集(但因为用了PCA，同一个数据降维的mapping, Ureduce 也需要对验证/测试集，使用！)



## 4. 无监督学习的所有输入数据<u>都是unlabeled</u>?

无监督学习，只是说 模型**上线后，用于实际业务**时，输入数据是unlabeled——在模型**训练/验证/测试**过程中，**也是要求**最优参数的(如，异常检测的异常阈值)，当然**可以使用labeled data**



## 5. 数据增强<u>器</u>对于<u>不同数据集</u>

已知给training set构造了ImageDataGenerator,  我们应该构造a <u>**separate**</u> generator for **valid and test sets**

### Why用构造不同的generator?

**Why can't we use the same generator as for the training data?**

LOk back at the generator we wrote for the training data.

- It normalizes each image **per batch**, meaning that it uses **batch statistics**.
- We should not do this with the test and validation data, since in a real life scenario we **don't process incoming images <u>a batch</u> at a time** (we process **<u>one image</u> at a time**).
- Knowing the average per batch of test data would effectively give our model an advantage.
  - The model should **<font color="#dd0000">not have any information about the test data</font>**.



### 但又该如何构造val/test set的数据增强generator呢？

What we need to do is **normalize** incoming test data using the <font color="#dd0000">statistics **computed from the training set**</font>.

There is one technical note. 

- Ideally, we would want to compute our **sample mean and standard deviation** using the **entire training set**. However, since this is **extremely large,** that would be very time consuming.

- In the interest of time, we'll **<font color="#dd0000">take a random sample of the dataset</font>** and do the calcualtion.

  

## 6. ml理论中常见的超平面概念

超平面一般化的广义叫法：

超平面是【分解平面】的一般化：

- 在一维的平面中，它是点 

- 在二维的平面中，它是线 

  - <font color="#dd0000">**为什么**</font>是平面？因为**normal vector法向量和plane是一一对应的**，等价！而实际正是normal vector在分割

    A plane would be this magenta line into two-dimensional space, and it **actually represents all the possible vectors that would be sitting on that line(之后都用plane).** In other words, they would be parallel to the plane, such as this blue vector or this orange vector. 

    You can define a plane with a single vector. This magenta vector is **perpendicular to the plane**, and it's called the **<u>normal vector</u> to that plane**. So normal vector is perpendicular to **any vectors that lie on the plane**. 

    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghqf03pey5j30ui0ki0vw.jpg" style="zoom:33%;" />

    - 关于分割：实际就是normal vector在发挥作用（而该vector对应它所垂直的plane！)

      如何在数学上而不是几何视觉上，计算分割的结果we are able to see **visually** when the vector is **on one side of the plane** or the other, but how do you **do this <u>mathematically</u>**?  把目标向量和normal vector，**做dot product**，正负号表明在同侧/异侧——实际是**俩向量，夹角的cos**在决定！！

      <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghqf6a99ekj31d60ig7c3.jpg" alt="2D plane，数学上是这样起到“分割”作用的" style="zoom:43%;" />

- 在三维的平面中，它是面 

- ...

- 在更高的维度中，我们称之为超平面** 
  所以，**广义上**，1D直线、2D平面，都可叫超平面



## 7. 区分点积(内积)、叉积(外积)、矩阵乘法

### a. 点积(内积): 结果是数(标量)

向量的点乘,也叫向量的内积、数量积，对两个向量执行点乘运算，就是对这两个向量对位①**一一相乘**之后 ②**再求和**的操作，点乘的结果是一个标量。



对于向量a和向量b，要求一维向量a和向量b的**行列数相同**，点积公式为：

![](https://img-blog.csdn.net/20160902214456788)





可以看出，计算结果是个标量！



#### 实现：numpy.dot 

注：虽然数学公式很清楚，但该np.dot要求了2个输入向量的维度，同**矩阵乘法**一样：**第一个v的行数 = 第二个v的列数**。

如：2个等长的向量v1, v2相乘——写成np.dot(v1, v2**.T**)



#### [几何意义](https://blog.csdn.net/dcrmg/article/details/52416832)



### b. 叉积(外积): 类似矩阵乘法

两个向量的叉乘，又叫向量积、外积、叉积，叉乘的运算结果**是一个向量**而不是一个标量。并且两个向量的叉积**与这两个向量组成的坐标平面垂直**。



对于向量a和向量b，叉乘公式为：

<img src="https://img-blog.csdn.net/20160902230539163" style="zoom:75%;" />

<img src="https://img-blog.csdn.net/20160902231520146" style="zoom:75%;" />

#### [几何意义](https://blog.csdn.net/dcrmg/article/details/52416832)



### 顺带一提：np.array的+-×÷

> The product operator * when used on arrays or matrices indicates **element-wise** multiplications.

**四则运算符号**，都是单纯的**<u>element-wise</u>的对位操作**，不像点积——不相加

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghh5wqiu4cj30m607241f.jpg" style="zoom:43%;" />

### **矩阵与矩阵/向量**相乘 **也可用<u>@</u>**

![也可用@！](https://tva1.sinaimg.cn/large/007S8ZIlgy1gj5p36wqcij31ji094gn0.jpg)

也自然要求：~~一维向量~~array a和b的**~~行列数~~<u>维度</u>相同**



## 8. 条件概率和交集概率的区分

### 交集概率: 分母是<u>全集</u>(所以默认忽略)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlntiq7ocj31dq0kak1t.jpg" style="zoom:30%;" />



### 条件概率: 分母是<u>做条件的那个概率</u>的所在集

#### 分子用了交集概率，分母是<u>做条件的概率</u>的所在集

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlnqr41w2j31f20jiaj3.jpg" style="zoom:30%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlnppukfmj315w0imaha.jpg" style="zoom:30%;" />



## 9. np.array的转置: 至少是2D

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho372sm9hj31f70u079y.jpg" style="zoom:40%;" />



## 10. 贝叶斯学习—ML的概率表示



<img src="/Users/kenny/Library/Application Support/typora-user-images/image-20200917171456223.png" style="zoom:43%;" />





一般来说一个<u>巨大（通常为无限）的假设空间</u>，可已有<u>无数个假设</u>。

贝叶斯学习的本质是：左边直观解释是，在**已有的(训练)数据**条件下，得到各个h的概率；公式右边分子是，**<u>符合(fit)这些数据D分布</u>的，假设h的概率**——学习的目标则是，确定左边那个 P(h|D) 的 argmax 函数，得到最优的h

为了简化，**去掉分母 P(D) 的项，因为它不依赖于假设而是已有的数据。这个方法被称为最大后验（MAP）。**

现在我们应用下面的数学技巧：

- 最大化原函数和最大化取对数的原函数的过程是相似的，即取对数不影响求最大值问题。
- 乘积的对数等于对数的和。
- 正量的最大化等同于负量的最小化。

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gitrm3yzidj30kr07274b.jpg" style="zoom:50%;" />



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gitrkb5ka4j31ew0agtbl.jpg" alt="总结" style="zoom:43%;" />

底数为 2 的负对数项看起来是不是很熟悉？这都来自「信息论」



## 11. 图像化表达：ML算法3种 VS 贝叶斯深度学习(神经网络)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gifuko31hlj31mf0u0b29.jpg" alt="" style="zoom:67%;" />



#### Bayesian 神经网络：让具体的标量参数，变为概率分布

贝叶斯深度学习的核心思想：将**神经网络的<u>权重w和b视作服从某分布的随机变量</u>**，而不是固定值。以及网络的前向传播，就是**<u>从权值分布中抽样</u>**然后计算。

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gifus5k4vuj30no15owp0.jpg" alt="Going Bayesian" style="zoom:30%;" />

进而，导致得到的**<u>model也不是唯一的</u>**，而是处于一个**<u>范围的所有</u>**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gifuqihxwsj30sa0fawhh.jpg" style="zoom:40%;" />

#### 贝叶斯学习的意义

**<u>提供不确定性</u>，非 softmax 生成的概率**。虽然工业界**一般需要确定性结果**，所以有这种贝叶斯机器学习在工业界没有广泛应用的错觉，但是贝叶斯理论**对于事件具有<u>很强的指导意义</u>**



## 12. 统计学 vs 统计(机器)学习

link: https://zhuanlan.zhihu.com/p/61964658

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gitsm0lryej31d00u0ans.jpg" alt="区分" style="zoom:67%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gitsff759nj30s40qw77x.jpg" alt="总结" style="zoom:40%;" />



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gitsfzesikj30yc0c842g.jpg" alt="总结" style="zoom:40%;" />



## 13. 急切学习(Eager)和惰性学习(Lazy)



### *(以下内容均为转载)*

急切学习方法也叫积极学习方法, 惰性学习方法也叫消极学习方法

### 急切/积极学习方法

指在利用算法**进行判断<u>之前</u>**,先利用训练集数据通过**训练得到一个目标函数**；在需要进行判断时，利用已经训练好的函数进行决策。

意义：这种方法是**在开始的时候需要进行一些工作,到后期进行使用的时候会很方便.**

**例如**

以很好理解的决策树为例,通过决策树进行判断之前,先通过对**训练集的训练建立起了一棵树**。

比如很经典的利用决策树判断一各新发现的物种是否为哺乳动物的例子,首先根据已经已知的训练集进行训练,提炼出哺乳动物的各种特征,这就是一个建立模型的过程；**模型建立好之后,再有新物种需要进行判断的时候**,只需要按照训练好的模型对照新物种挨个对比,就能很容易的给出判断结果.



### 惰性/消极学习方法

**最开始的时候不会根据已有的样本(通过"训练"过程)创建目标函数**,只是**简单的把训练用的样本<u>储存</u>好**；后期**需要对新进入的样本进行判断的时候**， 才开始**分析新进入样本与已存在的训练样本之间的关系**，并据此确定新入样本的目标函数值。

**例如**

典型算法**KNN**：KNN**不会根据训练集主动学习或者拟合出一个函数**，来对新进入的样本进行判断,而是单纯的记住训练集中所有的样本,并**没有像上边决策树那样先对训练集数据进行训练，<u>得出一套规则</u>**。所以它实际上**没有所谓的"训练"过程**,而是在**需要进行预测的时候从自己的训练集样本中查找**与新进入样本最相似的样本，来获得预测结果



### 对比：双方互补

积极学习方法：在训练时**考虑到了训练集中所有数据,训练时间比较长**；有新样本进入需要判断的时候**决策时间短**。但是也是由于这个原因,**<u>做增量拓展</u>的时候比较虐**.

消极学习方法：几乎没有训练时间,在新样本进入做判断的时候**计算开销大，时间长**；但是，**<u>天生支持增量学习</u>**。



那什么是增量学习？



## 14 增量学习(Incremental Learning)

**简单定义**：作为机器学习的一种方法，现阶段得到广泛的关注。在其中，**输入数据不断被用于扩展现有模型的知识，即进一步训练模型**。它代表了一种**动态**的学习的技术。

### 地位

是**<u>数据挖掘算法走向实用化</u>**的关键技术之一

### 启发

对于**传统的批量学习**技术来说,如何从**日益增加的新数据**中得到有用信息是一个难题。随着数据规模的不断增加,对时间和空间的需求也会迅速增加,最终会导致学习的速度赶不上数据更新的速度。机器学习是一个解决此问题的有效方法。

 随着人工智能和机器学习的发展,人们开发了很多机器学习算法。这些算法**大部分都是批量学习(Batch Learning)模式**,即**<u>需要在训练之前所有训练样本一次性可得</u>**,<font color="#dd0000">学习完这些样本**<u>之后</u>**——**学习过程就终止了,不再学习新的知识**</font>；然而在实际应用中,训练样本通常**<u>不可能一次全部得到,而是随着时间逐步得到</u>**,并且样本**反映的信息也可能随着时间产生了变化**。如果新样本到达后<u>**要重新学习**</u>全部数据,需要消耗大量时间和空间,因此批量的学习算法不能满足这种需求。

为了实现**在线学习**的需求,需要抛弃以前的学习结果,重新训练和学习,这对时间和空间的需求都很高。因此,迫切需要研究增量学习方法,可以**渐进的进行知识更新**,且能**修正和加强以前的知识**,使得更新后的知识能**适应**新增加的数据。

### 优势

增量学习是指一个学习系统能**<u>不断地从新样本中学习新的知识,并能保存大部分以前已经学习到的知识</u>**。增量学习非常<font color="#dd0000">**类似于人类自身的学习模式**</font>。因为人在成长过程中,每天学习和接收新的事物,学习是逐步进行的,而且,对已经学习到的知识,**人类一般是不会遗忘的**。

与传统的数据分类技术相比，增量学习分类技术具有显著的优越性，这主要表现在两个方面：一方面由于其**无需保存历史数据**，从而减少存储空间的占用；另一方面，由于其在新的训练中**充分<u>利用了历史的训练结果</u>，从而显著地减少了后续训练的时间**。

增量学习技术是一种得到广泛应用的**<u>智能化数据挖掘与知识</u>**发现技术。其思想是当样本逐步积累时，学习精度也要随之提高。

### 特点

对于满足以下条件的学习方法可以定义为增量学习方法：

- 可以学习**新的**信息中的有用信息
- **不需要访问**已经用于训练分类器的**原始数据**
- 但是，对已经学习的知识具有**记忆**
- 在面对新数据中包含的新类别时，可以有效地进行处理



## 15 No free lunch理论