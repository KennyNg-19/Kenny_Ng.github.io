---
title: DeepLearning.ai出品NLP的Course1-NLP的Classification & Vector Spaces
date: 2020-08-08 13:45:24
img: https://miro.medium.com/max/2344/1*uc2HNS1m4CjG8Yb4AxGqbQ.png
tags: [NLP, math]
---



## Course1: NLP with Classification and Vector Spaces

## 1. Task: text classification



### 法1: Logsitc回归模型





### 法2: 纯靠词频的Naive Bayes模型

> Naive Bayes is an example of **supervised machine learnin**g, and shares **many similarities with the logistic regression** method 

#### <font color="#dd0000">Why Naive? 单纯地靠词频 → 概率</font>

this method makes the assumption that **the features(比如  句中的词前后是有关系的，或者说有某些词总是常见伴随出现的，这些相关性会影响词频) you're using for classification are <u>all independent</u>**, which in reality is **rarely the case**.



##### (不足)Naive 的2个理想前提

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmww5z78ej30z607kq58.jpg" alt="2个assumption" style="zoom:33%;" />

###### 1. 忽视本身多个词的关联性

Some words **①often appear together** and/or they might also **②be related to the thing they're always describing**. 



These words in a sentence are not always necessarily independent of one another, but **Naive Bayes assumes that they are**. This could lead you to **<font color="#dd0000">under or over estimates the conditional probabilities of individual word</font>——一般把常常一起出现的词，当做互相独立，更多是<u>underestimate</u>它们的条件概率**. Naive Bayes **might assign equal probability to all words** even though **from the context** you can see that one of them **is the most likely candidate**. 



<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmx4bziloj31a207qjxg.jpg" style="zoom:50%;" />



###### 2. 各个类比例不均匀分布的原始数据集 distribution of the training data sets



A good data set will **contain the same proportion of positive and negative tweets** as **<u>a random sample would</u>**. Most of the available annotated corpora are <u>**artificially**</u> balanced just like the data set you'll use for the assignment. 



Howver, in the real tweet stream, positive tweet is sent to occur **more often** than their negative counterparts. One reason for this is that negative tweets might contain content that is banned by the platform or muted by the user such as inappropriate or offensive vocabulary. Assuming that reality behaves as your training corpus, this could **also result in a very optimistic or very pessimistic model——一样影响了prevalence p(pos) p(neg)，进而影响模型**



##### 优势: 简单快捷

但Naive Bayes依然可以用于**简单的分类问题**

It takes a **short time** to train and also has a short prediction time.



------



先说结果：

#### Training pipeline 5步

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlr7qj6mxj30yg0u04hx.jpg" alt="朴素贝叶斯pipeline" style="zoom:43%;" />



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlr8wg82vj311i0di46n.jpg" alt = "training过程总结" style="zoom:30%;" />

#### 学习步骤: 本质是naive的 频率 → 概率

Step1 同逻辑斯特回归，计算词频

Step2 根据词频，计算在各类中的**<u>条件概率</u>** 

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpxj3b6wj31eo0n6n93.jpg" alt="根据词频计算条件概率" style="zoom:33%;" />



Step3 **同一个**词的正负类**比例相除**，再**各个词的相乘**——最后得到**偏向类**，**以1位界限**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlq1b1z5qj31gk0ia14g.jpg"  style="zoom:40%;" />

比例为1，则为neutral；大于1，则偏向正类；......



##### 改进1: Laplacian Smoothing 平滑处理-避免出现概率为0

###### 背景:为什么要做平滑处理?

　　**零概率问题**：在计算实例的概率时，如果某个量x，**只是因为<u>在观察样本库（训练集）中</u>没有出现过，它的频率为0，会导致整个实例的概率结果是0**。

具体的如，在文本分类的问题中，当一个词语没有在训练样本中出现，该词语调概率为0，使用连乘计算文本出现概率时也为0。

这是不合理的，**不能因为一个事件，<u>在当前数据集中</u>没有观察到，就武断的认为该事件的概率就是0**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlq010yc8j30le0hcwl8.jpg" alt="当词频(概率)为0" style="zoom:33%;" />

回到NLP，因为naive公式，要各个词在各个类的条件概率的比值，相乘——但如果出现**频率为0**的词，一样会导致乘法结果无意义

**[Laplacian Smoothing](https://dcpnonstop.github.io/2017/11/24/%E5%B9%B3%E6%BB%91%E5%A4%84%E7%90%86-%E6%8B%89%E6%99%AE%E6%8B%89%E6%96%AF%EF%BC%88laplace%EF%BC%89/)(拉布拉斯平滑处理，又称加1平滑)**相当于加了个bias，让**概率转化为接近0的数 而避免了0**。这种转化对于结果影响自然很小



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlovcgbq2j30x608odhw.jpg" style="zoom:33%;" />

###### Smoothing公式

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlowop7gij31gu0jy7e8.jpg" style="zoom:33%;" />



##### Naive Bayes <u>Inference</u>

###### ratio定义

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlp0hfq3zj31fi0n0aoy.jpg" style="zoom:33%;" />



ratio的别名：**likelihood**



###### 题外话：Prior ratio先验分布 ——有用，尤其当数据集是unbalanced的

<font color="#dd0000">Why prior ?</font>

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlt1seunwj31oo0ce0vk.jpg" style="zoom:53%;" />

如果数据本身unbalanced，则Naive Bayes公式前面必须多乘一项，先验分布的比例！

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpcysq7tj30pe0bsgpp.jpg" alt="所以正规朴素贝叶斯形式应该如下" style="zoom:40%;" />



课程内的样本数据集是理想的，均匀分布：I haven't mentioned it till now because in this small example, we have **exactly the same number of** positive and negative tweets, making the ratio one. In this week's assignments, you'll have a balanced datasets, so you'll be working with a ratio of one. 

> In the future though, when you're building your own application, remember that **this term becomes important for unbalanced datasets.** 



##### 改进2: log likelihood-概率太小数了，取对数方便计算

> Carrying out **small number multiplications** runs the risk of **numerical underflow** when the number returned is so small it can't be stored on the device

累程 变 **累加**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpi83j76j31gs0dejy7.jpg" alt="用log-likelihood ratio后的朴素贝叶斯公式" style="zoom:33%;" />

我们将log-likelihood ratio，新定义为**λ**



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpns2a6mj31h80g2gyg.jpg" alt="0就是界限，为neutral" style="zoom:33%;" />

然后再对**各个λ求和**，如：大于0，则该句子情感为pos类...

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlpqjrj4nj31hm0eigw7.jpg" style="zoom:33%;" />



好处1: **重新定义分类界线——从1改成0**

即neg类概率更大时 ratio可以为负，负数更直观



好处2: 原始ratio的区间<u>长度并不对称</u>，neg类只能<font color="#dd0000">取值[0,1)，neg sentiment程度不明显</font>！用了log就是<u>**长度完全对称的区间**</u>！

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlptv6i4ij31ca0je46t.jpg" alt="区间的改变" style="zoom:33%;" />



#### Testing: 用于predict



##### 如果测试时，出现模型之前没见到过词，就当neutral！

> The values that don't show up in the table **are considered neutral** and don't contribute anything to this score. The **ML model can only give a score for words it's seen before.**



别忘了用了log累加时，如果数据不平很 最后得加上prior的log！

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlrtq4ujyj31h00msh1e.jpg" alt="interview这个词没学过，则为neutral" style="zoom:33%;" />



##### 总结

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghlryf207cj317g0myk0s.jpg" style="zoom:33%;" />



#### Naive Bayes的延伸分类应用

上面是bayes公式，上下相除**抵消P(tweet)**, 可得下面的公式

下面是之前bayes训练的公式(本身就有 **P(w/pos)的累乘 = P(tweet/pos)**)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmvic7kvmj30rk0ec0yt.jpg" alt="该分类模型延伸应用的公式推导" style="zoom:53%;" />

下面左边，就是**模型预测的结果(ratio)**——<font color="#dd0000">这种形式，可以延伸应用到**其他2分类领域**</font>



<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmvje5q7nj312c0hi0ym.jpg" alt="延伸领域" style="zoom:35%;" />

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmwqw0z0ej313k0k6aq1.jpg" alt="消除二分歧义" style="zoom:33%;" />



#### 单靠<u>词频</u>的Naive Bayes的潜在error

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmxep6f9cj30ui0buacj.jpg" alt="image-20200811155634232" style="zoom:50%;" />

##### 1. <u>semantic meaning lost</u> in the pre-processing step

教训：还是务必查看原始句子语义，而不是单纯的移除标点符号和stop words

##### 移除特殊标点(构成了表情，决定了情感)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmxj90vh5j30r80a6adu.jpg" alt="表情符号不可移除！" style="zoom:33%;" />



The sad face punctuation in this case is very **important to the sentiment**  because it tells you what's happening. 

But if you're removing punctuation, then the processed tweet will leave behind only  beloved grandmother, which looks like a very positive tweet. 



##### 移除stop words

not这二元180°转义词，**也是stop word！**

 <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghmxn45612j31ea0860z7.jpg" style="zoom:33%;" />

If you remove neutral words like not and this, what you're left with is the [Good, attitude, close, nice]

From this set of words, any classifier will infer that this is something very positive.



##### 2. word order affects the meaning of a sentence

word order自然是语义的一部分

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmxor6jelj312k0ciwmh.jpg" style="zoom:33%;" />



##### 3. quirks(怪癖) of languages come naturally to humans but <u>confuse</u> models.

Quirk： 人类语言中的 带有sarcasm irony讽刺、euphemism委婉等色彩，adversarial(**恰恰反义、敌对**)性质词

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmxqfa2p7j31fo0he4ab.jpg" alt="这种反义，属于adversarial" style="zoom:33%;" />

------



## 2. Word Vector 和 Vector Space入门

词向量，向量空间模型

### Why vector space? 形式和应用





#### 形式

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmylqhg13j30m805edhp.jpg" style="zoom:43%;" />



#### 应用(底层)

Vector space models will 



1. 验证句子语义的相似性：identify whether the first pair of questions or the second pair **are similar in meaning** even if they do not share the same word 即**identify similarity** for a question answering, paraphrasing, and summarization.
   
   

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghmyakwvu0j319a0j6gw0.jpg"  style="zoom:33%;" />



2. <font color="#dd0000">*发掘语言中，词之间**关联性**</font>: to **<font color="#dd0000">capture dependencies between words</font>**——应用很广！！！



<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghmyi6ab4oj31e40iuk6o.jpg" alt="这个可以引用到很多领域，常见如下" style="zoom:45%;" />



### 共现矩阵



#### word by word的matrix



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghmz1fgbtgj31ea0f8n6b.jpg" alt="若词出现在k距离内次数越频繁，则这组词越相关" style="zoom:33%;" />



#### word by doc的



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghnzcssgqnj313s0kgdqr.jpg" alt="在doc库的各种类，出现的频率" style="zoom:33%;" />



#### 引出Vector Space

这些**<font color="#dd0000">有了维度</font>的**数据，就可以<font color="#dd0000">**放入vector space**， 进行相似度分析</font>

然后可以看成N-维向量(N为词的数量)，通过比较**向量的”相似性“指标，如距离**——得出句子/语料的相似性**



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghnzi90xuuj31ei0ligyp.jpg" style="zoom:30%;" />



##### vector space的应用：挖掘word analogies

**infer unknown relations** among words



###### 如通过词已知的关系，推导出，<u>词之间相似但未知的关系</u>

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho0me9qq5j30zy0jgwn9.jpg" alt="推测未知的首都" style="zoom:33%;" />





<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho0nh4gpxj31hk0j6ajs.jpg" alt="用向量减法；再在[10,4]处找，相似指标最大的向量" style="zoom:33%;" />





##### 指标1：Euclidean distance 欧几里得距离❌

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghnzmivjmhj31ck0kgn7q.jpg" style="zoom:40%;" />



Generalize到更高维的

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghnzny5azlj313m06ygnh.jpg" alt="该公式也是向量的norm，向量/矩阵的长度或大小" style="zoom:40%;" />



##### 指标2：cosine similarity✅



欧几里得距离的缺点：明显**会受到corpus size的影响**——**导致vector size**长短不一，影响判断

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghnzy5kjw3j31g20nqwsh.jpg" alt="food和农业关联更大，但因food语料库太小，欧几里得误判history关系更大" style="zoom:50%;" />





**cosine similarity**

即向量的**点积**公式:   多考虑了**语料的大小，即向量长度**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho04e04nij30n80b8wgg.jpg" alt="点积" style="zoom:33%;" />

这和向量的”相似度“**有何关联**？

 

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho07dveoxj31fe0hwn4x.jpg" alt="0-90°的夹角，也对相似度有影响" style="zoom:33%;" />



#### <font color="#dd0000">word embeddings 词嵌入</font>(向量)

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghp2rgex4zj31p005mwh8.jpg" alt="word represents by vector" style="zoom:53%;" />

....



------

#### PCA: 高纬数据的降维

a **statistical technique**

##### 目的: 方便可视化

降维到2D方便可视化，来找关系

**Word embeaddings** end up having **vectors in very, very high dimensions**.



PCA：a way to **reduce the dimension of these vectors to two dimensions so you can plot it on an X-Y axis, 2D plot.** 

helpful for <u>**visualizing**</u> your data to check if your representation is **<u>capturing relationships</u> among words**

![](https://tva1.sinaimg.cn/large/007S8ZIlly1ghp7blx33uj31po092adg.jpg)



##### 步骤

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho4f0u4nrj31hu0j2drh.jpg" style="zoom:33%;" />



需要：向量的**特征值和特征向量**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho6suirinj317u074goy.jpg" style="zoom:33%;" />



为什么要不相关的feature? 因为**自然语言文本总是上下文相关的**，所以feature之间总有一定相关性，

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho6ula7l9j31io0ne12y.jpg" alt="Step1 生成uncorrelated features" style="zoom:33%;" />

注：PCA works better **if the data is <u>centered</u>**



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gho6veh9o8j31cu0guwo4.jpg" alt="Step2 通过和特征向量的点积，求出压缩后的向量" style="zoom:33%;" />

------



## 3. 词向量Task: 机器翻译和相似doc搜索

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghp8c3q9suj31bw0h2n5y.jpg" alt="相关知识" style="zoom:50%;" />



### Task: 翻译

[一种实现：维特比算法](https://kennyng-19.github.io/Kenny_Ng.github.io/2020/01/29/NLP-intro/#1-case-%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91)



另一种实现：**已知两种语言的word embeddings**，通过**寻找<u>transform matrix</u>**，找到转换后 **目标语言中最相似的word vector**——我们称该过程为**"align word vectors"**



#### Step1: Transfor vector by <u>matrix</u>

find **a transformation matrix** from English to French vector space embeddings. 

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghp9x50omjj31e40hadnt.jpg" alt ="align word vectors" style="zoom:33%;" />



Such a transformation matrix is **a matrix that <font color="#dd0000">rotates and scales vector spaces</font>**——回忆《**线性代数的本质**》

##### **然后怎么计算该矩阵呢？还是优化问题**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghpa0ho7c8j30ya0gk78h.jpg" alt="还是梯度下降，这不过这次是矩阵的" style="zoom:50%;" />



##### 补充notation-Frobenius范数: 即矩阵元素的平方和的开方

F范数是**针对矩阵而言**的，具体定义可以**类比向量的L2范数**

<img src="https://pic3.zhimg.com/80/ded8e501d6d54eaf3d218491423380df_1440w.jpg?source=1940ef5c" alt="F范数" style="zoom:80%;" />

当然在ML中，为了简化计算(因为范数只是用于**<u>最优化</u>**)，可以**不开方而是保留平方**



所以保留开方后，Loss函数最终形式为：

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghpa22ro8jj30z00b876y.jpg" alt="Loss函数最终形式" style="zoom:33%;" />





拓展，**3 main vector transformations的几何意义**

(更多，请回忆《**线性代数的本质**》)

- Scaling
- Translatio
- Rotation

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghp9qtx6x8j313f0u0dr9.jpg" alt="关于施加rotation矩阵的结论" style="zoom:50%;" />



##### Step 2: 寻找最相似的<u>几个</u>翻译结果by <u>KNN</u>

因为word embedding空间不一定有，和matrix转换的**结果数值一模一样的词向量**，且存在<u>近义词</u>——所以一般是会输出**几个最近似**的词向量，供选择。这里会用到KNN算法

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghpa5okk64j314g0fm0zw.jpg" style="zoom:30%;" />

##### 改进版 faster <u>approximate</u> KNN

##### 启发思路：空间划分

(下图只是**简单的2D空间**) **slice the space <u>into regions</u>**: you could **search just <u>within</u> those regions**. When you think about organizing subsets of a dataset efficiently, you may think about placing your data **into <u>buckets</u>**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghqbl9ocp8j31220jywwe.jpg" style="zoom:33%;" />



If you think about buckets, then you'll definitely want to think about <font color="#dd0000">**<u>hash tables</u>**</font>.



###### 提升KNN，处理<u>高维数据</u>的效率：Locality Sensitive Hashing

本质是 a **hash function**, to be locality sensitive; an **[algorithmic technique](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)** that hashes **similar input items into the same "buckets" with <u>high probability</u>**——所以说是一种**近似法, "approximate"**

白话定义：把vector根据在**vector space中的距离足够近**的**分到一起**，的Hashing方法

> Locality is another word for location, sensitive is another word for caring 
>
> This is kNN in simple terms: You have a labelled dataset and now you are trying to label a new data point. Find the k nearest data points from your labelled dataset to the new point. The majority vote among the k nearest neighbors is the label of the new point. Add the new point and it's label to your dataset
>
> One of the **biggest problems with kNN 处理高维数据时** is that **常规的暴力法下，for each new data point, you have to calculate its distance from all existing points in your dataset.** The LSH technique, differing from [conventional hashing techniques](https://en.wikipedia.org/wiki/Hash_function) in that hash collisions are maximized, not minimized,  can be seen as a way to **reduce the dimensionality of high-dimensional data**; high-dimensional input items can be **reduced to low-dimensional versions while preserving <u>relative distances</u> between items**. And this problem is what LSH is **trying to solve**.



So locality sensitive hashing is a **hashing method** that's **cares very deeply about assigning items based on where they're <u>located in vector space</u>**.



###### 核心：<u>**Multiplanes hash functions**</u>

> In order to divide your vector space **into <u>manageable</u> regions**, you'll want to use **<u>more than one plane</u>**. Based on the idea of **numbering every single region** that is **formed by the <u>intersection of n planes</u>.**

[思路](https://kennyng-19.github.io/Kenny_Ng.github.io/2020/07/25/ML-concepts-memorize/#6-ml理论中常见的超平面概念)：每一个plane，实际就是定义一个**法向量**

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghrkcqi8smj313602kjsl.jpg" style="zoom:33%;" />

几何上，you have multiple planes and it helps us to divide the vector space into smaller sub regions. But you **want to have a single hash value** to know **which bucket to assign the vectoring**. You do this by **combining the signals from all the planes** into a single hash value.



那么定义一组plane就等于一组法向量 output value is a **combination of the side of the plane** where the vector is localized with respect to the collection of planes.

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghrkdjszw6j30zu0lygq0.jpg" style="zoom:33%;" />



Locality Sensitive Hashing**最终计算公式**：看sign定boolean值h，再用2的幂次求和公式

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghrkhg1by3j312a0k0k0i.jpg" style="zoom:40%;" />



###### 注意: 因为是随机生成的法向量-plane，请重复多次得到更合理的结果 make sets of **random planes** 

You will make <u>multiple</u> sets of **<u>random planes</u>** in order to make the approximate nearest neighbors **more accurate.**



### Task: 相似doc搜索

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghp89bzp7ej31e00bgq9h.jpg" style="zoom:33%;" />

同理用fast KNN



虽然doc的表示和word的vector表示不完全相同，但doc也是word组成——可以用word embedding中存在的word vector值的**累加**，表示

<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghqhe1yls3j30zy0my7ai.jpg" alt="累加word vector的embedding值" style="zoom:33%;" />



<img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ghqhihbgq4j310k0gw4br.jpg" alt="拆分doc成word, 用embedding中存在的word vector累加" style="zoom:33%;" />

有了所有doc的vector，剩下的寻找**近似目标**，就和上述KNN过程一样了...

