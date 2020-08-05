---
title: NLP-text-process
date: 2020-06-07 15:06:46
tags: [NLP]
---



# NLP文本处理

## NLP general pipeline

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjqjhtaemj31kj0u0qte.jpg" alt="pipeline" style="zoom:30%;" />

## 文本处理阶段

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjqm2tae6j31go08gai2.jpg" alt="text process" style="zoom:30%;" />



### 常见处理环节



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfm2bbdbwoj30zg0eaaj2.jpg" style="zoom:33%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghg5ar7dblj30pi06kdgk.jpg" alt="预处理文本大致步骤" style="zoom:50%;" />



------



## A. 分词的常用底层算法

所有分词的工具，都要**<u>依赖于词库</u>**

### 1. 基于<u>匹配</u>规则的方法

#### <u>最大</u>匹配算法 Max Matching——<font color="#dd0000"><u>贪心</u></font>算法

##### 何为<u>最大</u>？(人为设的)每次截取的最大长度——<u>经验</u>决定，看词典和这种语言中<u>词语常见的最大长度</u>

- <u>前向</u>最大匹配(forward-max matching)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjs4728c6j31gi0t07wh.jpg" style="zoom:30%;" />

- <u>后向</u>最大匹配(backward-max matching): 句子的**遍历方向，从右到左**和前向**相反**而已
  - 和前向的效果90%大多数情况，结果一样
  - 不一样的情况：后向可能会导致一些分词的琐碎

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjs9uae0oj31cg0nudwn.jpg" style="zoom:30%;" />

- 双向：结合了前向和后向，双向遍历



##### 该算法的缺点

- 贪心只能找局部最优，未必最优 

- 效率，依赖于max_length

- <font color="#dd0000">容易歧义，因为算法不支持考虑**<u>语义</u>**</font>，**不智能**。导致机器学习的：**纯粹是单词**，没有任何**句法**甚至**语义**

  - 比如一个case：在词典中，一个词是另一个词的子集，而最大匹配会**保留那个更长的词**(**匹配到**该更长的词，则提前结束遍历)

    <img src="/Users/kenny/Library/Application Support/typora-user-images/image-20200607161233659.png" style="zoom:35%;" />



<hr>

那么，什么样的算法工具，能考虑到语义？想想**语言模型**(机器翻译也用到了)

语言模型，可以evaluate一句话<font color="#dd0000">**语义的正确性**</font>——基于**已有的词频的统计**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjt7ifbxlj310m0doak1.jpg" alt="人工的词频统计" style="zoom:33%;" />

所以，对于分词结果，计算其语义合理性

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjt8qkr98j319209cn88.jpg" style="zoom:33%;" />



### 2. 基于概率统计的方法——考虑语义

#### by<u>语言模型</u>(<u>机器翻译</u>也用到了)



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjux5rp84j31760u07wh.jpg" style="zoom:30%;" />



##### 可能的问题和解决: underflow 

问题：词频统计的概率太小，小数一旦再乘积，导致underflow的溢出

解决：**数学处理技巧**，给概率加上log**转化为<u>累加</u>**，利用log的**<u>递增型</u>**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfjtgdx5m3j316y0f6ws9.jpg" alt="log转化" style="zoom:33%;" />



##### 缺点和解决

##### 缺点：如果要考虑所有可能，复杂度太高

##### 解决：维特比算法

(问题和解决方法，都和我[NLP-intro](https://kennyng-19.github.io/Kenny_Ng.github.io/2020/01/29/NLP-intro/)文中的机器翻译case的**穷举法下**的情况，**是一样的**)



#### 用维特比算法来优化

维特比**基于DP动态规划**

//TODO ........





## B. 拼写纠错



### 编辑距离(edit dis.)

#### 3种编辑操作

replace, add和delete。

#### 需要操作的数量=edit dis



### 算法目标：找到<u>编辑距离较小</u>的候选words

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfm4rhshbsj31fo0ra7f0.jpg" style="zoom:30%;" />

#### ❌低效率的暴力<u>搜索</u>法

遍历词典的**所有词**，逐一比较

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfm4ot9eayj31ma0qodvp.jpg" style="zoom:27%;" />



#### ✅升级版: <u>生成</u> 距离较小的字符串

##### step1: 生成大量words

**当然，生成**也要用到3种常规**编辑操作**



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfm4vjv2trj30bq0lotet.jpg" style="zoom:33%;" />

###### 问题: 为什么只生成dis为1，2的？

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfm4widswij309s05idh7.jpg" style="zoom:25%;" />

因为经验主义，**大多数情况**，**人为的**单词拼写错误，只是1、2个字母的错误



##### Step2: 过滤筛选

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfm544piarj30ry0a8q6z.jpg" style="zoom:50%;" />

###### 问题定义，转为条件概率问题

c，为**现有词典里的<font color="#dd0000">正确的</font>单词**

现在要，求出每一个**生成的word，s** 其<font color="#dd0000">**对应<u>所有</u>可能的c中**</font>概率最大的那个c ——即为**"最正确"**的

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfm55fyha3j3152090wmp.jpg" style="zoom:33%;" />



基于贝叶斯定理，

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfmgp0y1e1j319e0daan3.jpg" style="zoom:37%;" />



再细看，理解**P(s|c)和P(c)**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfmgqyuhkdj30wk0660yy.jpg" style="zoom:50%;" />

理解P(s|c)，**基于统计**：人写错单词c为s的概率

举例: 人把apple写成其他形式s，如app, appl的概率

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfmjmfoqeqj30bg0g479w.jpg" style="zoom:50%;" />

理解P(c)：即**基于统计的词频**，单词c(在语料库)出现的概率



## C. 移除停用词Stop words

### 移除对象

通常移除的是**停用词(对意义理解影响很小的)、出现频率很低(noise, outlier偏离主语义的)**的词

注意：虽然有的停用词很常见，但也要**考虑应用场景**考虑**是否移除**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfn8q0eqdxj30rg058tcj.jpg" alt="image-20200610154719443" style="zoom:33%;" />

比如：”好，很好“这种停用词，在<u>sentiment analysis</u>中就<u>不应该被移除</u>——反而可能需要**修改**<u>停用词库</u>



#### 例子

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghg591mrcpj31ko0sutgr.jpg" style="zoom:33%;" />



### 注意: 在情感分析中，stem的词可只留着前缀！

因为**分类**问题，只需要知道什么词占对应情感类别**的比例/程度**即可

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghg5d0rmhjj31ie0au40q.jpg" alt="stem后 词并不是正确拼写的" style="zoom:43%;" />

## D. 词的标准化(英文文本)

### 标准化的对象

英文中常用的词语的多种形式(单复数 时态等)，在**语义理解**上该统一

<img src="/Users/kenny/Library/Application Support/typora-user-images/image-20200610160111641.png" style="zoom:35%;" />

### 常用技术

#### Stemming: 基本还原成原形

> Stemming is the process of converting a word **to its most general form, or stem**. This helps in reducing the size of our vocabulary.

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghg0nir63dj31340gm43m.jpg" style="zoom:33%;" />



#### Lemmazation(更严格)



### stemmer底层原理实例

需要<u>语言学家</u>(懂词根词缀的)来制定！

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gfn93dbta0j31pw0po7wh.jpg" style="zoom:33%;" />