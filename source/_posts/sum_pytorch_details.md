---
title: Pytorch一些小细节补充
date: 2019-08-26 16:32:47
tags: [ML]
---



# 补充Pytorch中的易被忽视但得注意的细节

#### 1. 定义Dataloader的num_workers参数

**Q**: 

在给Dataloader设置worker数量（`num_worker`）时，到底**设置多少合适**？这个worker到底怎么工作的？
**如果将`num_worker`设为0（也是默认值），就没有worker了吗？**



**A**: `num_workers`的经验设置值**是<font color="#dd0000">自己电脑/服务器的CPU核心数</font>**，(比如我的Macbook Pro16寸是 6核)如果CPU很强+配的**RAM是充足的**，就可以设置为**略微≥核心数**。



1. 参数的**作用**： 官方注释为：

   >  how many **subprocesses** to use for data loading.
   >
   >  ``0`` means that the data will be loaded **in the main process.** (default: ``0``)

   作用：denotes the **<font color="#dd0000">number of processes that generate batches in parallel</font>** to generate your data <font color="#dd0000">**on multiple cores** in real time</font>

   详细解释：每每轮到dataloader加载数据时：

   ```python
   for epoch in range(start_epoch, end_epoch):
   		for i, data in enumerate(trainloader):
   ```

   dataloader**一次性创建`num_worker`个worker**，（也可以说dataloader一次性创建`num_worker`个工作进程，worker也是普通的工作进程），并用`batch_sampler`将指定batch分配给指定worker，worker将它负责的batch加载进RAM。

   然后，dataloader从RAM中找本轮迭代要用的batch，如果找到了，就使用。如果没找到，就要`num_worker`个worker继续加载batch到内存，直到dataloader在RAM中找到目标batch。一般情况下都是能找到的，因为`batch_sampler`指定batch时当然优先指定本轮要用的batch。

2. **不同赋值时各自的意义**：`num_worker`设置得大，好处是**寻batch速度快**，因为下一轮迭代的batch很可能在上一轮/上上一轮...迭代时已经加载好了。坏处是**内存开销大**，也加重了CPU负担（worker加载数据到RAM的进程是CPU复制的嘛）。

   如果设为0，意味着每一轮迭代时，dataloader**不再有自主加载数据到RAM**这一步骤（因为没有worker了），而是在RAM中找batch，找不到时再加载相应的batch。缺点当然是**速度更慢**。

   



#### 2. 