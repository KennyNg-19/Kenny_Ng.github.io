---
title: NLP-text-feature-represent
date: 2020-06-10 16:28:09
tags: [NLP]

---



# 文本的表示和特征提取



## 文本表示(向量)

### 单词

#### one-hot

**维度=|词典|**

<img src="/Users/kenny/Library/Application Support/typora-user-images/image-20200610163125511.png" style="zoom:33%;" />



### 句子

#### 1. <u>Sparse</u> boolean representation(类似one-hot)

**维度=|词典|**

##### 不管词频

**无论**一个单词出现了**几次**，**都设为1**(true)



##### 明显缺点

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghg07t9frvj31ay0m2aiz.jpg" alt="参数量 ∝ 单词总量V" style="zoom:33%;" />





#### 2. Count-based representation(记词频)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfna64u3v3j31g80aiaqo.jpg" alt="image-20200610163726467" style="zoom:33%;" />



### 句子相似度

**欧氏距离**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfnfjibpxsj30j00980yn.jpg" style="zoom:33%;" />

