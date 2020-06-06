---
title: (转)算法复杂度-intro(推导分治算法的master theorem)
date: 2018-02-06 16:10:13
tags: [算法]
---

在实现算法的时候，通常会从两方面考虑算法的复杂度，即时间复杂度和空间复杂度。

顾名思义，时间复杂度用于度量算法的计算工作量；空间复杂度用于度量算法占用的内存空间。

由于算法测试结果非常**依赖测试环境，且受数据规模的影响，**因此需要一种粗略的评估方法，即时间、空间复杂度分析方法。



## 渐进时间复杂度

理论上来说，时间复杂度是算法运算所消耗的时间，但是对于一个算法来说，评估运行时间是很难的，因为针对不同大小的输入数据，算法处理所要消耗的时间是不同的，因此通常关注的是时间频度，即算法运行计算操作的次数，记为 ![[公式]](https://www.zhihu.com/equation?tex=T%28n%29) ，其中n称为问题的规模。同样，因为n是一个变量，n发生变化时，时间频度 ![[公式]](https://www.zhihu.com/equation?tex=T%28n%29) 也在发生变化，我们称时间复杂度的极限情形称为算法的“渐近时间复杂度”，记为 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%29) 。这种表示方法称作**大 O 时间复杂度表示法**。大 O 时间复杂度实际上并不具体表示代码真正的执行时间，而是表示**代码执行时间随数据规模增长的变化趋势**，因为我们没办法用通常的评估方式来评估所有算法的时间复杂度，所以通常使用渐进时间复杂度表示算法的时间复杂度，下文中的时间复杂度均表示渐进时间复杂度。

我们在这里放一个python算法的例子来解释一下：

```python3
def f(n):
	a,b=0,0#运行消耗t0时间
	for i in range(n):#运行一次平均消耗t1时间
		a = a + rand()#运行一次平均消耗t2时间
	for j in range(n):#运行一次平均消耗t3时间
		b = b + rand()#运行一次平均消耗t4时间
```

在这个例子中，我们分别计算 ![[公式]](https://www.zhihu.com/equation?tex=f%28n%29) 函数的时间复杂度与空间复杂度。根据代码上执行的平均时间假设，计算出来执行 ![[公式]](https://www.zhihu.com/equation?tex=f%28n%29) 的时间为 ![[公式]](https://www.zhihu.com/equation?tex=T%28n%29%3Dt_0%2B%28t_1%2Bt_2%29%2An%2B%28t_3%2Bt_4%29%2An%3Dt_0%2B%28t_1%2Bt_2%2Bt_3%2Bt_4%29%2An) ；而函数中申请了两个变量a,b，占用内存空间为2。上述 ![[公式]](https://www.zhihu.com/equation?tex=T%28n%29) 是我们对函数f(n)进行的准确时间复杂度的计算。但实际情况中，输入规模n是影响算法执行时间的因素之一。在n固定的情况下，不同的输入序列也会影响其执行时间。当n值非常大的时候， ![[公式]](https://www.zhihu.com/equation?tex=T%28n%29) 函数中常数项t0以及n的系数 ![[公式]](https://www.zhihu.com/equation?tex=%28t_1%2Bt_2%2Bt_3%2Bt_4%29) 对n的影响也可以忽略不计了，因此这里函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28n%29) 的时间复杂度我们可以表示为 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%29) 。我们再来看空间复杂度，跟时间复杂度表示类似，也可以用极限的方式来表示空间复杂度（但是貌似没有渐进空间复杂度的说法），因为这里只声明了2个变量，因此空间复杂度也是常数阶的，因此这里空间复杂度计算为 ![[公式]](https://www.zhihu.com/equation?tex=O%281%29) 

**规律总结：**

**所有代码的执行时间 T(n) 与每行代码的执行次数 n 成正比。**

![[公式]](https://www.zhihu.com/equation?tex=T%28n%29%3DO%28f%28n%29%29)

其中f(n) 表示每行代码执行的次数总和。公式中的 O，表示代码的执行时间 ![[公式]](https://www.zhihu.com/equation?tex=T%28n%29) 与 ![[公式]](https://www.zhihu.com/equation?tex=+f%28n%29+) 表达式成正比。



### 计算方法

因为我们计算的是极限状态下的时间复杂度，因此存在两种特性：

1.按照函数数量级角度来说，相对增长低的项对相对增长高的项产生的影响很小，可忽略不计。

2.最高项系数对最高项的影响也很小，因此也可以忽略不计。针对第1点，常见的时间复杂度有：常数阶：常数阶： ![[公式]](https://www.zhihu.com/equation?tex=O%281%29) , 对数阶： ![[公式]](https://www.zhihu.com/equation?tex=O%28log_2+n%29) , 线性阶： ![[公式]](https://www.zhihu.com/equation?tex=O%28n%29) , k次方阶： ![[公式]](https://www.zhihu.com/equation?tex=O%28n%5EK%29) ,指数阶： ![[公式]](https://www.zhihu.com/equation?tex=O%282%5En%29) 。

根据上述两种特性，总结时间复杂度的计算方法：

1.**加法法则-总复杂度等于量级最大的那段代码的复杂度：**计算时只取相对增长最高的项，去掉低阶项，并去掉最高项的系数（比如 ![[公式]](https://www.zhihu.com/equation?tex=O%282n%29) 只需要表示为 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%29) ）；

2.**只关注循环执行次数最多的一段代码**：大 O 这种复杂度表示方法只是表示一种变化趋势。通常会忽略掉公式中的常量、低阶、系数，只需要记录一个最大阶的量级就可以了。所以，我们在分析一个算法、一段代码的时间复杂度的时候，也只关注循环执行次数最多的那一段代码就可以了；

3.**乘法法则-嵌套代码的复杂度等于嵌套内外代码复杂度的乘积：**如果 ![[公式]](https://www.zhihu.com/equation?tex=T_1%28n%29%3DO%28f%28n%29%29%2CT_2%28n%29%3DO%28f%28n%29%29) ,那么 ![[公式]](https://www.zhihu.com/equation?tex=T%28n%29%3DT_1%28n%29%E2%88%97T_2%28n%29%3DO%28f%28n%29%29%E2%88%97O%28g%28n%29%29%3DO%28f%28n%29%E2%88%97g%28n%29%29.) 也就是说假设![[公式]](https://www.zhihu.com/equation?tex=+T_1%28n%29+%3D+O%28n%29%EF%BC%8CT_2%28n%29+%3D+O%28n%5E2%29) ，则 ![[公式]](https://www.zhihu.com/equation?tex=T_1%28n%29+%2A+T_2%28n%29+%3D+O%28n%5E3%29)

4.**针对常数阶，取时间复杂度为O(1)**。





## 空间复杂度

空间复杂度指的是算法在内存上临时占用的空间，包括程序代码所占用的空间，输入数据占用的空间和变量占用的空间。在递归运算时，由于递归计算是需要使用堆栈的，所以需要考虑堆栈操作占用的内存空间大小。空间复杂度的计算也遵循渐进原则，即参考时间复杂度与空间复杂度计算方法项。

### 计算方法

1.通常只考虑参数表中为形参分配的存储空间和为函数体中定义的局部变量分配的存储空间（比如变量 ![[公式]](https://www.zhihu.com/equation?tex=a%3D0) 在算法中空间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%281%29) ； ![[公式]](https://www.zhihu.com/equation?tex=list_a%3D%5B0%2C1%2C....%2Cn%5D) 的空间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%29) ； ![[公式]](https://www.zhihu.com/equation?tex=set%28list_a%29) 的空间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%281%29) )。

2.递归函数情况下，空间复杂度等于一次递归调用分配的临时存储空间的大小乘被调用的次数。

我们这里以递归方法实现的斐波那契数列为例：

```python3
def fib(n):
    if n < 3: 
        return 1
    else:
        return fib(n-2)+fib(n-1)
```

斐波那契数列的序列依次为 ![[公式]](https://www.zhihu.com/equation?tex=1%2C1%2C2%2C3%2C5%2C8%2C13.....) 特点是，当数列的长度大于等于3时，数列中任意位置的元素值等于该元素前两位元素之和。

1.计算时间复杂度： ![[公式]](https://www.zhihu.com/equation?tex=O%282%5En%29) 。计算方法：通过归纳证明的方法，我们尝试计算数列中第8个元素的值的递归调用次数,为了方便观察，我把外层括号替换为了大括号。 ![[公式]](https://www.zhihu.com/equation?tex=fib%288%29%3Dfib%287%29%2Bfib%286%29+%3D%7Bfib%286%29%2Bfib%285%29%7D%2B%7Bfib%285%29%2Bfib%284%29%7D+) ![[公式]](https://www.zhihu.com/equation?tex=%3D%28%7Bfib%285%29%2Bfib%284%29%7D%2B%7Bfib%284%29%2Bfib%283%29%7D%29%2B%28%7Bfib%284%29%2Bfib%283%29%7D%2B%7Bfib%283%29%2Bfib%282%29%7D%29+%3D.....) 这里太多我就不一一写出了。不难发现，每次调用递归时，递归调用次数都是以程序中调用递归次数即2的指数形式增长的。第一层递归时，调用了2次 ![[公式]](https://www.zhihu.com/equation?tex=fib%28n%29) 函数；第二层递归时，第一层的2次递归调用分别又要调用2次，即调用了 ![[公式]](https://www.zhihu.com/equation?tex=2%5E2) 次；第三层递归调用了 ![[公式]](https://www.zhihu.com/equation?tex=2%5E3) 次，以此规律，不难算出时间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%282%5En%29)

2.计算空间复杂度： ![[公式]](https://www.zhihu.com/equation?tex=O%28n%29) 。计算方法：我们同样使用归纳证明的方法，尝试推导数列中第6个元素值的内存占用情况。调用函数 ![[公式]](https://www.zhihu.com/equation?tex=fib%286%29) ,此时因为有形参n传递，在栈中为n申请内存资源，我们为了方便，以 ![[公式]](https://www.zhihu.com/equation?tex=fib%286%29) 表示栈中元素。此时栈中有 ![[公式]](https://www.zhihu.com/equation?tex=fib%286%29) ;我们根据函数内的递归调用关系，为了计算 ![[公式]](https://www.zhihu.com/equation?tex=fib%286%29) ,我们需要 ![[公式]](https://www.zhihu.com/equation?tex=fib%285%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=fib%284%29) 的值，此时发生形参传递，栈中有 ![[公式]](https://www.zhihu.com/equation?tex=fib%286%29%2Cfib%285%29%2Cfib%284%29) ;为了计算 ![[公式]](https://www.zhihu.com/equation?tex=fib%284%29) ，我们需要 ![[公式]](https://www.zhihu.com/equation?tex=fib%283%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=fib%282%29) 的值，此时栈中有 ![[公式]](https://www.zhihu.com/equation?tex=fib%286%29%2Cfib%285%29%2Cfib%284%29%2Cfib%283%29%2Cfib%282%29) ，但是由于 ![[公式]](https://www.zhihu.com/equation?tex=fib%282%29%3D1) ，此时 ![[公式]](https://www.zhihu.com/equation?tex=fib%282%29) 函数计算完成， ![[公式]](https://www.zhihu.com/equation?tex=fib%282%29) 出栈，此时栈中有 ![[公式]](https://www.zhihu.com/equation?tex=fib%286%29%2Cfib%285%29%2Cfib%284%29%2Cfib%283%29) 。为了计算 ![[公式]](https://www.zhihu.com/equation?tex=fib%283%29) ，需要 ![[公式]](https://www.zhihu.com/equation?tex=fib%282%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=fib%281%29) 的值，此时栈中有 ![[公式]](https://www.zhihu.com/equation?tex=fib%286%29%2Cfib%285%29%2Cfib%284%29%2Cfib%283%29%2Cfib%282%29%2Cfib%281%29) 。但是 ![[公式]](https://www.zhihu.com/equation?tex=fib%282%29%3D1%2Cfib%281%29%3D1) ,计算完成， ![[公式]](https://www.zhihu.com/equation?tex=fib%282%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=fib%281%29) 出栈。此时得到fib(3)的值为2，fib(3)出栈；由此出栈顺序， ![[公式]](https://www.zhihu.com/equation?tex=fib%284%29%2Cfib%285%29%2Cfib%286%29) 也会随计算完成出栈。不难发现，在此次递归计算的过程中，内存中最多消耗了6个内存资源，由归纳证明法得出 ![[公式]](https://www.zhihu.com/equation?tex=fib%28n%29) 的空间复杂度为 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%29) 。



## Master Theorem 主方法

Master Theorem是为了计算**<font color="#dd0000">含有递归调用**的**分治**算法</font>的时间复杂度的。因为算法中如果含有递归调用算法的情况下**使用归纳证明的方法**，计算时间复杂度是相当困难的，因此需要利用**已有的定理Master Theorem**来帮我们计算复杂情况下的算法的时间复杂度



### 定理的定义

![img](https://pic1.zhimg.com/80/v2-d05e32c23bdbbd70ef3dea46eb47db58_1440w.jpg)

Master Theorem的一般形式是T(n) = a T(n / b) + f(n)， a >= 1, b > 1。递归项f(n)理解为一个高度为log_b n 的a叉树， 这样总时间频次为 (a ^ log_b n) - 1， 右边的f(n)假设为 nc 那么我们对比一下这两项就会发现 T(n)的复杂度主要取决于 log_b a 与 f(n) 的大小。（log_b a表示以b为底，a的对数）因此我总结了使用Master Theorem的三种case的简单判断：

1.计算 ![[公式]](https://www.zhihu.com/equation?tex=log_b+a) 的值，比较 n^(log_b a)与f(n)的大小。

2.若n^(log_b a)>f(n),时间复杂度为O(n^(log_b a)) （case 1）

3.若n^(log_b a)<f(n),时间复杂度为O(f(n)) （case 3）

4.若n^(log_b a)=f(n),时间复杂度为O(n^(log_b a)*(log n)^k+1) （case 2） (其中k值为f(n)中如果有log对数项时，该对数项的指数为k，例如，如果f(n)=log n ，k=1；f(n)=n,k=0)

可能公式理解起来有点困难，举几个例子来加深理解：

![img](https://pic4.zhimg.com/80/v2-40b032e934608df08aabbbd79d1b18cb_1440w.jpg)

### 总结

- 常规套路，就是**比较n^(log_b a)与f(n)**的值了，比较出来就可以套用上述公式

- 但是一定要注意公式中的限制条件，如a必须为常数项等。



Reference:

《数据结构与算法之美》

[The Master theorem](http://people.csail.mit.edu/thies/6.046-web/master.pdf)