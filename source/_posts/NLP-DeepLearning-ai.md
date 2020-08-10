---
title: DeepLearning.ai新课-NLP系列 要点
date: 2020-08-08 13:45:24
tags: [NLP]
---



# 1. Task: text classification



## 法1: Logsitc回归





## 法2: Naive Bayes

> Naive Bayes is an example of **supervised machine learnin**g, and shares **many similarities with the logistic regression** method 

### <font color="#dd0000">Why Naive?</font>

this method makes the assumption that **the features(比如 词频，句中的词前后是有关系的，或者说有某些词总是常见伴随出现的) you're using for classification are <u>all independent</u>**, which in reality is **rarely the case**.



但依然可以用于**<u>简单的分类</u>**问题

 it still works nicely as a **simple method for sentiment analysis.**

#### 优势: 简单快捷

It takes a **short time** to train and also has a short prediction time.



先说结果：

### Training pipeline 5步

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlr7qj6mxj30yg0u04hx.jpg" alt="朴素贝叶斯pipeline" style="zoom:43%;" />



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlr8wg82vj311i0di46n.jpg" alt = "training过程总结" style="zoom:30%;" />

### Naive Bayes学习步骤

Step1 同逻辑斯特回归，计算词频

Step2 根据词频，计算在各类中的**<u>条件概率</u>** 

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpxj3b6wj31eo0n6n93.jpg" alt="根据词频计算条件概率" style="zoom:33%;" />



Step3 **同一个**词的正负类**比例相除**，再**各个词的相乘**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlq1b1z5qj31gk0ia14g.jpg"  style="zoom:40%;" />

比例为1，则为neutral；大于1，则偏向正类；......



#### 改进1: Laplacian Smoothing-避免出现概率为0

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlq010yc8j30le0hcwl8.jpg" alt="当词频(概率)为0" style="zoom:33%;" />

因为naive公式，需要各个词在各个类的条件概率的比值，相乘——所以如果存在**频率为0**的词，会导致乘法结果无意义

Laplacian Smoothing相当于加了个bias，让**概率转化为接近0的数 而避免了0**。这种转化对于结果影响自然很小



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlovcgbq2j30x608odhw.jpg" style="zoom:33%;" />

##### Smoothing公式

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlowop7gij31gu0jy7e8.jpg" style="zoom:33%;" />



#### Naive Bayes <u>Inference</u>

##### ratio定义

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlp0hfq3zj31fi0n0aoy.jpg" style="zoom:33%;" />

#### 

ratio的别名：**likelihood**



##### 题外话：Prior ratio先验分布 ——有用，尤其当数据集是unbalanced的

###### <font color="#dd0000">Why prior ?</font>

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlt1seunwj31oo0ce0vk.jpg" style="zoom:53%;" />

如果数据本身unbalanced，则Naive Bayes公式前面必须多乘一项，先验分布的比例！

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpcysq7tj30pe0bsgpp.jpg" alt="所以正规朴素贝叶斯形式应该如下" style="zoom:40%;" />



课程内的样本数据集是理想的，均匀分布：I haven't mentioned it till now because in this small example, we have **exactly the same number of** positive and negative tweets, making the ratio one. In this week's assignments, you'll have a balanced datasets, so you'll be working with a ratio of one. 

> In the future though, when you're building your own application, remember that **this term becomes important for unbalanced datasets.** 



#### 改进2: log likelihood-概率太小数了，取对数方便计算

> Carrying out **small number multiplications** runs the risk of **numerical underflow** when the number returned is so small it can't be stored on the device

累程 变 **累加**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpi83j76j31gs0dejy7.jpg" alt="用log-likelihood ratio后的朴素贝叶斯公式" style="zoom:33%;" />

我们将log-likelihood ratio，新定义为**λ**

##### 

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpns2a6mj31h80g2gyg.jpg" alt="0就是界限，为neutral" style="zoom:33%;" />

然后再对**各个λ求和**，如：大于0，则该句子情感为pos类...

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpqjrj4nj31hm0eigw7.jpg" alt="image-20200810144535440" style="zoom:33%;" />



##### 好处1: 重新定义分类界线：从1改成0，即neg类概率更大时 ratio可以为负

##### 好处2: 原始ratio的区间<u>长度并不对称</u>，neg类只能<font color="#dd0000">取值[0,1)，neg sentiment程度不明显</font>！用了log就是<u>长度完全对称的区间</u>！

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlptv6i4ij31ca0je46t.jpg" alt="区间的改变" style="zoom:33%;" />



### Testing: 用于predict



#### 如果测试时，出现模型之前没见到过词，就当neutral！

> The values that don't show up in the table **are considered neutral** and don't contribute anything to this score. The **ML model can only give a score for words it's seen before.**



别忘了用了log累加时，如果数据不平很 最后得加上prior的log！

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlrtq4ujyj31h00msh1e.jpg" alt="interview这个词没学过，则为neutral" style="zoom:33%;" />



#### 总结

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlryf207cj317g0myk0s.jpg" style="zoom:33%;" />