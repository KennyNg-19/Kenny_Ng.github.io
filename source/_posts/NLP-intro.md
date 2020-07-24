---
title: NLP-intro
date: 2020-01-29 13:30:38
tags: [ML, NLP]
---



## 0. NLP Overview

### a. 按照<u>研究</u>方向~语言的I/O

2个大方向对应了，语言的**"一进一出"**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gf99ljg3ooj30v80hswh7.jpg" style="zoom:35%;" />

### b. 按照<u><font color="#dd0000">现实中的业务</font></u>大致分成2大类

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0xhkrae0j31440jwna9.jpg" alt="2大类常见业务" style="zoom:35%;" />

#### 文本生成任务: **文本序列→文本序列**

比如机器翻译，文本风格迁移等，

#### 类别识别任务: 文本序列→类别

比如情感分类，实体命名识别，主题分类，槽位填充等。

##### 输入：多个序列

多个序列要怎么办呢？大概两种可能的解法。

一是把两个序列，分别用**两个模型**去做编码。再把它们编码后的嵌入，丢给另一个整合的模块，去得到最终的输出。有时，我们也会在两个模型之间加 Attention，确保二者的编码内容能互相意识。

二是，近年来比较流行的做法是直接**把两个句子连接起来，中间加一个特殊的字符**，如 <u>**BERT**</u> 里面的 <SEP>，来提示模型去意识到这是两个句子的分隔符(接起来的序列丢给模型后可以预测下游任务)



#### 其他: 这2大类的变体

- **Word Segmentations 分词**

  英文词汇有**空格符**分割词的边界，但有的语言，如**中文，却没有类似的方式来区分**，所以需要**额外的分词处理**。在一个句子中找出词的边界有时并不是一个简单的问题，所以也需要模型来做。

  <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0xppt5gbj311r0u0wpj.jpg" style="zoom:25%;" />

- **POS Tagging 词性标注**: Part of Speech **词性**分析(词性是很重要的特征)

   需要标记出一个句子中的每个词的词性是什么。对应输入一个序列，输出序列每个位置的**类别任务**。

  <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0xotv23zj31020jsthy.jpg" style="zoom:33%;" />

- **Pre-Processing work** 自然语言理解的**前处理**的任务

  - **Parsing 句法分析**

    给定句子，根据<u>词性</u>生成**树状结构** by CYK算法

  - **Dependency Parsing**(词的)依存分析 

    生成**graph**

    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0y22yxs4j31200ncwll.jpg" style="zoom:33%;" />

  - **Coreference Resolution 指代消解**

    模型需要把输入文章中**指代同样东西的部分，找出来**。比如当He 和 Paul Allen 指的就是同一个人。

- **Summarization 生成摘要**

  分成2种

  - 过去常用的是**抽取式摘要 Extractive**

    把一篇文档看作是许多句子的组成的序列，模型需要从中找出最能熔炼文章大意的句子提取出来作为输出。它相当于是对每个句子做一个二分类，来决定它要不要放入摘要中。但仅仅把每个句子分开来考虑是不够的。我们需要模型输入整篇文章后，再决定哪个句子更重要。这个序列的基本单位是一个句子的表征

    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0xx1vlv0j314a0rkqcu.jpg" style="zoom:33%;" />

  - 近年流行的**生成式摘要 Abstractive**

    模型的输入是一段长文本，输出是短文本。输出的短文本往往会与输出的长文本**有很多共用的词汇**。这就需要模型在生成的过程中有, 把文章中重要词汇拷贝出来再放到输出中的, 复制能力，比如 Pointer Network。

    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0xy8wzhhj313y0pgaio.jpg" style="zoom:33%;" />

- **NER：Named Entity Recognition 命名实体识别**——**<u>抽取</u>**语句中我**重点关注**的**名词**

  并没有非常清楚的定义：它取决于我们**对哪些事情关注**，随着领域的不同，有所差异，取决于我们的**具体应用场景**。一般的实体包括人名、组织和地名等等。

  比如想让机器读大量医学相关的文献，希望它自动知道有什么药物可以治疗新冠状肺炎。<u>这些药物的名字，就是实体</u>。它输入的是一个序列，输出的是序列上每个位置的类别。它就和词性标注、槽位填充一样。

  NER常见的两个问题: **名字一样但指的是不同的东西**，有多个标签需要实体消歧；**不一样的名字指的却是相同的东西**，需要实体归一化。

  <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0y4ykt6uj312m0regxp.jpg" style="zoom:33%;" />

- **Relation Extraction**(重要): 关系抽取

  假如已知如何从文本中获得实体，接下来还需要知道**它们之间的关系**。

  关系抽取的输入：**序列和抽取出的实体**，输出是**两两实体之间的关系**。本质是一个**分类任务**

  比如，输入：哈利波特和霍格沃茨，输出：关系，是后者的学生

  <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0yb0p0i0j314g0t8nd9.jpg" style="zoom:33%;" />

- **Question Answering** 问答系统/搜索引擎

- **Chating** 对话机器人

  分成两种，闲聊和任务导向型。

  - **Chatbox** 闲聊机器人：基本上都是在尬聊，有一堆问题待解决，比如角色一致性，多轮会话，对上下文保有记忆等。

  - **Task-oriented** 任务导向的对话机器人：能**协助人完成某件事**

    比如订机票，调闹钟，问天气等。我们需要一个模型，**把过去已经有的历史对话，统统都输入到一个模型中**，这个模型**可以输出一个序列**当作现在机器的**回复**

    <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0ym6grvaj313a0peauo.jpg" style="zoom:33%;" />

    - 我们会把这个模型**再细分成很多模块**，而不会是端对端的

      这些模块通常包括：**自然语言理解NLU，行动策略管理，自然语言生成NLG**

      - 自然语言理解负责根据上下文去**理解当前用户的意图**，

        方便

      - 行动策略(policy)管理**选出下一步候选的行为**。如执行系统操作，澄清还是补全信息，

        确定好行动之后，

      - 自然语言生成模块会生成出对齐行动的回复

      - (除了之前三个模块，再加上语音助手的<u>语音识别 ASR 和 语音合成 TTS</u> 就成了完整的对话系统。)

      <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gh0yqezmcfj314w0rktk0.jpg" style="zoom:33%;" />

    

### c. 技术的4个(实现难度递增的)维度

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gff3scbnfmj31980pynao.jpg" alt="从下往上难度递增" style="zoom:33%;" />





## 1. case-机器翻译

### 思路a: 暴力法

#### 原理：**全排列组合**分词后的句子，用语言模型从中筛选出语义最合适的



第一步通过**翻译模型TM**进行**分词**

第二步映射词语，因为语法的存在而进行<u>映射后词语的排列组合</u>

第三步通过**语言模型LM**选出排列组合中最合适的语句

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf99soj3vmj317w0kuq8d.jpg)缺点: 排列组合的复杂度太高

解决: **维特比Viterbi**算法——**同时考虑TM**分词+**LM筛选**这分开2个步骤



### 思路b: **维特比Viterbi**算法

#### 原理：贝叶斯TM和LM

**串联TM**分词+**LM筛选**这分开的2个步骤



#### 流程

![流程图](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9i9qd3wlj31680f20vt.jpg)

各个部分的作用

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9iup3gr2j30x20icwjg.jpg)好处：降低了复杂度，**避免了原来随机排列组合生成句子**的**NP-hard的指数级**复杂度

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9iao2h3fj309802e749.jpg)



##### 1. 语言模型LM

##### 作用：生成句子(即输出)

##### 分类：按照当前生成的语言，对之前词语的”记忆“程度

这种**连续的条件概率**——又称，**联合分布**

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9i640a7bj316e0aojut.jpg)

**Unigram** model, **Bingram** model .... n-gram model

至于每个Prop, 则源于**已有的<u>统计频率</u>结果**



###### 性能标准

生成之后同时看**句子质量**和**对应概率(score)**，句子**错误少**的**概率应该越大**

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gf9iknu2qsj30yw06idh3.jpg)

##### 2. 翻译模型TM

...



## 2. case-基于检索的智能问答系统

### 数据/输入: 语料库

![写好的问与答结对](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfinf6trm9j311w0kuke0.jpg)

### NLP general pipeline

![image-20200607150135228](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjqjhtaemj31kj0u0qte.jpg)



####  QAsys的流程

![image-20200606183943480](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfir8926c2j31ip0u0e81.jpg)

#### 解决：如何搜索出最相似问题

##### 相似度的衡量？

- ❌**逐字**比较：正则/**规则**AI——**对匹配要求很高很高**, 所以需要考虑**<u>尽可能多</u>**的输入，过于繁琐
  - 唯一使用场景：**<u>没有数据时</u>**，语料库很空
- ✅**概率**比较：字符串的**"相似性"计算**





#### 阶段一 [文本处理](https://kennyng-19.github.io/Kenny_Ng.github.io/2019/08/07/NLP-text-process/)



常用技术

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfm2bbdbwoj30zg0eaaj2.jpg" alt="image-20200609152001651" style="zoom:50%;" />