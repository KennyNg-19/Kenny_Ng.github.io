---
title: NLP-DeepLearning.ai
date: 2020-08-10 13:45:24
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



### Naive公式



#### 技巧: Laplacian Smoothing-避免出现概率为0

因为naive公式，需要各个词在各个类的条件概率的比值，相乘——所以如果存在概率为0的词，会导致最后结果无意义

Laplacian Smoothing相当于加了个bias，让**概率转化为接近0的数 而避免了0**。这种转化对于结果影响自然很小