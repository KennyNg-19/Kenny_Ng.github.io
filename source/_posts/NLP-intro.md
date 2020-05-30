---
title: NLP-intro
date: 2020-02-29 13:30:38
tags: [ML, NLP]
---



## 0. NLP的研究方向

2个大方向对应了，语言的**"一进一出"**

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf99ljg3ooj30v80hswh7.jpg)

## 1. NLU之机器翻译

## 思路a: 暴力法

### 原理：**全排列组合**分词后的句子，用语言模型从中筛选出语义最合适的



第一步通过**翻译模型TM**进行**分词**

第二步映射词语，因为语法的存在而进行<u>映射后词语的排列组合</u>

第三步通过**语言模型LM**选出排列组合中最合适的语句

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf99soj3vmj317w0kuq8d.jpg)缺点: 排列组合的复杂度太高

解决: **维特比Viterbi**算法——**同时考虑TM**分词+**LM筛选**这分开2个步骤



## 思路b: **维特比Viterbi**算法

### 原理：贝叶斯TM和LM

**串联TM**分词+**LM筛选**这分开的2个步骤



### 流程

![流程图](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9i9qd3wlj31680f20vt.jpg)

各个部分的作用

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9iup3gr2j30x20icwjg.jpg)好处：降低了复杂度，**避免了原来随机排列组合生成句子**的**NP-hard的指数级**复杂度

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9iao2h3fj309802e749.jpg)



#### 1. 语言模型LM

##### 作用：生成句子(即输出)

##### 分类：按照当前生成的语言，对之前词语的”记忆“程度

这种**连续的条件概率**——又称，**联合分布**

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9i640a7bj316e0aojut.jpg)

Unigram model, Bingram model .... n-gram model

至于每个Prop, 则¨源于提前的计算(概率统计)



##### 性能标准

生成之后同时看Prop，句子**错误少**的**概率应该越大**

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9iknu2qsj30yw06idh3.jpg)

#### 2. 翻译模型TM

###### 