---
title: NLP-intro
date: 2020-01-29 13:30:38
tags: [ML, NLP]
---



# 0. NLP Overview

## a. NLP研究方向

2个大方向对应了，语言的**"一进一出"**

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf99ljg3ooj30v80hswh7.jpg)

## b. 技术的4个维度

![IMG_6B342F1B480B](https://tva1.sinaimg.cn/large/007S8ZIlgy1gff3scbnfmj31980pynao.jpg)

### 简称-全称/英文

- **Relation Extraction**(重要): **关系抽取**

- Parsing：句法分析——给定句子，根据<u>词性</u>生成**树状结构** by CYK算法

- Dependency Parsing： (词的)依存分析 ——生成**graph**



- POS: Part of Speech **词性**分析(词性是很重要的特征)

- NER：Named Entity Recognition **命名实体**识别——**<u>抽取</u>**语句中，我**重点关注**的**名词**



# 1. app-机器翻译

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

生成之后同时看**句子质量**和**对应概率(score)**，句子**错误少**的**概率应该越大**

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9iknu2qsj30yw06idh3.jpg)

#### 2. 翻译模型TM

...



# 2. app-基于检索的智能问答系统

### 数据/输入: 语料库

![写好的问与答结对](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfinf6trm9j311w0kuke0.jpg)

### 流程

![image-20200606183943480](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfir8926c2j31ip0u0e81.jpg)



### 解决：如何搜索出最相似问题

#### 相似度的衡量？

- ❌**逐字**比较：正则/**规则**AI——**对匹配要求很高很高**, 所以需要考虑**<u>尽可能多</u>**的输入，过于繁琐
  - 唯一使用场景：**<u>没有数据时</u>**，语料库很空
- ✅**概率**比较：字符串的**"相似性"计算**



